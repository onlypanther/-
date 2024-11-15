from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session, g
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import sqlite3
from werkzeug.utils import secure_filename
from datetime import datetime
from functools import wraps
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'tu_clave_secreta_aqui'

# Configuración
DATABASE = 'database.db'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'jfif'}

# Crear carpetas necesarias
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Usando dispositivo: {device}")

# Transformaciones optimizadas
transform_predict = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Funciones de base de datos
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE, detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row
    return g.db

def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    if not os.path.exists(DATABASE):
        db = get_db()
        with app.open_resource('schema.sql') as f:
            db.executescript(f.read().decode('utf8'))

# Inicializar la base de datos
with app.app_context():
    init_db()

# Decorador para requerir login
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Cargar modelo optimizado
try:
    model = models.resnet34(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Linear(256, 2)  # Solo 2 clases: bueno y malo
    )
    model = model.to(device)
    
    checkpoint = torch.load('mejor_modelo.pth', map_location=device)
    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Modelo cargado exitosamente")
except Exception as e:
    logger.error(f"Error al cargar el modelo: {str(e)}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def verificar_calidad_imagen(imagen):
    if imagen.size[0] < 100 or imagen.size[1] < 100:
        return False, "La imagen es demasiado pequeña. Tamaño mínimo requerido: 100x100 píxeles"
    
    extremos = imagen.getextrema()
    if all(max(channel) - min(channel) < 50 for channel in extremos):
        return False, "La imagen tiene poco contraste. Por favor, use una imagen más clara"
    
    return True, None

def procesar_imagen(imagen):
    try:
        es_valida, mensaje_error = verificar_calidad_imagen(imagen)
        if not es_valida:
            return {'error': mensaje_error}

        if imagen.mode != 'RGB':
            imagen = imagen.convert('RGB')
        
        imagen_tensor = transform_predict(imagen).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(imagen_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            prob_bueno = round(probabilities[0][0].item() * 100, 2)
            prob_malo = round(probabilities[0][1].item() * 100, 2)
            
            logger.info(f"Probabilidades: Bueno={prob_bueno}%, Malo={prob_malo}%")
            
            # Clasificación con umbral ajustado
            if prob_bueno > 52:  # Favorece ligeramente "bueno"
                return {
                    'prediccion': 'bueno',
                    'probabilidad_bueno': prob_bueno,
                    'probabilidad_malo': prob_malo,
                    'mensaje': 'El electrodoméstico está en buen estado'
                }
            else:
                return {
                    'prediccion': 'malo',
                    'probabilidad_bueno': prob_bueno,
                    'probabilidad_malo': prob_malo,
                    'mensaje': 'El electrodoméstico está en mal estado'
                }
            
    except Exception as e:
        logger.error(f"Error al procesar imagen: {str(e)}")
        return {'error': f'Error al procesar la imagen: {str(e)}'}

def save_scan_result(user_id, result, prob_good=0.0, prob_bad=0.0):
    db = get_db()
    try:
        db.execute('''
            INSERT INTO history (user_id, result, probability_good, probability_bad, date)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (user_id, result, prob_good, prob_bad))

        db.execute('''
            UPDATE statistics 
            SET total_scans = total_scans + 1,
                {} = {} + 1,
                last_updated = CURRENT_TIMESTAMP
            WHERE user_id = ?
        '''.format(
            'good_results' if result == 'bueno' else 'bad_results',
            'good_results' if result == 'bueno' else 'bad_results'
        ), (user_id,))
            
        db.commit()
        logger.info(f"Resultado guardado exitosamente para usuario {user_id}")
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error al guardar resultado: {str(e)}")
        raise

def get_history(user_id):
    db = get_db()
    return db.execute('''
        SELECT 
            strftime('%d/%m/%Y %H:%M:%S', date, 'localtime') as date,
            result,
            probability_good,
            probability_bad
        FROM history 
        WHERE user_id = ?
        ORDER BY date DESC
        LIMIT 10
    ''', (user_id,)).fetchall()

def get_statistics(user_id):
    db = get_db()
    stats = db.execute('''
        SELECT 
            total_scans,
            good_results,
            bad_results,
            last_updated,
            ROUND(CAST(good_results AS FLOAT) * 100 / NULLIF(total_scans, 0), 1) as good_percentage,
            ROUND(CAST(bad_results AS FLOAT) * 100 / NULLIF(total_scans, 0), 1) as bad_percentage
        FROM statistics 
        WHERE user_id = ?
    ''', (user_id,)).fetchone()
    
    if stats is None:
        db.execute('INSERT INTO statistics (user_id) VALUES (?)', (user_id,))
        db.commit()
        return {
            'total_scans': 0,
            'good_results': 0,
            'bad_results': 0,
            'last_updated': datetime.now(),
            'good_percentage': 0.0,
            'bad_percentage': 0.0
        }
    
    return stats

# Rutas
@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('home'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        error = None
        
        user = get_db().execute(
            'SELECT * FROM users WHERE username = ?', (username,)
        ).fetchone()

        if user is None:
            error = 'Usuario incorrecto.'
        elif not password == user['password']:
            error = 'Contraseña incorrecta.'

        if error is None:
            session.clear()
            session['user_id'] = user['id']
            return redirect(url_for('home'))

        flash(error)

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        error = None

        if not username:
            error = 'Se requiere nombre de usuario.'
        elif not password:
            error = 'Se requiere contraseña.'
        elif db.execute(
            'SELECT id FROM users WHERE username = ?', (username,)
        ).fetchone() is not None:
            error = f'El usuario {username} ya está registrado.'

        if error is None:
            db.execute(
                'INSERT INTO users (username, password) VALUES (?, ?)',
                (username, password)
            )
            db.commit()
            
            user_id = db.execute('SELECT last_insert_rowid()').fetchone()[0]
            db.execute('INSERT INTO statistics (user_id) VALUES (?)', (user_id,))
            db.commit()
            
            flash('Registro exitoso. Por favor inicia sesión.')
            return redirect(url_for('login'))

        flash(error)

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/home')
@login_required
def home():
    return render_template('index.html')

@app.route('/estadisticas')
@login_required
def estadisticas():
    return render_template('estadisticas.html', stats=get_statistics(session['user_id']))

@app.route('/historial')
@login_required
def historial():
    return render_template('historial.html', history=get_history(session['user_id']))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No se encontró ningún archivo'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'})
    
    if file and allowed_file(file.filename):
        try:
            resultado = procesar_imagen(Image.open(file))
            
            if not resultado.get('error'):
                save_scan_result(
                    session['user_id'],
                    resultado['prediccion'],
                    resultado['probabilidad_bueno'],
                    resultado['probabilidad_malo']
                )
            
            return jsonify(resultado)
                
        except Exception as e:
            logger.error(f"Error en predict: {str(e)}")
            return jsonify({'error': f'Error al procesar la imagen: {str(e)}'})
    
    return jsonify({'error': 'Tipo de archivo no permitido'})

@app.teardown_appcontext
def close_db_at_end_of_requests(e=None):
    close_db(e)

if __name__ == '__main__':
    try:
        logger.info("Iniciando servidor Flask...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logger.error(f"Error al iniciar el servidor: {str(e)}")
        exit(1)