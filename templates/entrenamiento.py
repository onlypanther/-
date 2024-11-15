import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import logging
from torch.utils.data.sampler import WeightedRandomSampler

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Usando dispositivo: {device}")

class ElectrodomesticosDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        try:
            # Cargar imágenes buenas (etiqueta 0)
            bueno_dir = os.path.join(root_dir, 'bueno')
            if os.path.exists(bueno_dir):
                for img_name in os.listdir(bueno_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.jfif')):
                        self.images.append(os.path.join(bueno_dir, img_name))
                        self.labels.append(0)
            else:
                logger.error(f"No se encontró el directorio: {bueno_dir}")
            
            # Cargar imágenes malas (etiqueta 1)
            malo_dir = os.path.join(root_dir, 'malo')
            if os.path.exists(malo_dir):
                for img_name in os.listdir(malo_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.jfif')):
                        self.images.append(os.path.join(malo_dir, img_name))
                        self.labels.append(1)
            else:
                logger.error(f"No se encontró el directorio: {malo_dir}")
            
            logger.info(f"Total imágenes buenas: {self.labels.count(0)}")
            logger.info(f"Total imágenes malas: {self.labels.count(1)}")
            
            if len(self.images) == 0:
                raise ValueError("No se encontraron imágenes para el entrenamiento")
                
        except Exception as e:
            logger.error(f"Error al cargar las imágenes: {str(e)}")
            raise

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img_path = self.images[idx]
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            logger.error(f"Error al cargar la imagen {img_path}: {str(e)}")
            raise

def train_model():
    try:
        # Configuración
        num_epochs = 50  # Aumentado para mejor aprendizaje
        batch_size = 8   # Batch size pequeño para dataset pequeño
        learning_rate = 0.0002  # Learning rate bajo para mejor precisión
        
        # Transformaciones mejoradas
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(20),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=20,
                translate=(0.15, 0.15),
                scale=(0.8, 1.2)
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Crear dataset
        logger.info("Cargando dataset...")
        dataset = ElectrodomesticosDataset('datos_entrenamiento', transform=transform)
        
        # Calcular pesos para balancear clases
        class_counts = [dataset.labels.count(i) for i in range(2)]
        total_samples = sum(class_counts)
        
        # Pesos inversamente proporcionales al número de muestras
        weights = torch.FloatTensor([
            total_samples / (2 * class_counts[0]),  # peso para buenas
            total_samples / (2 * class_counts[1])   # peso para malas
        ])
        
        # Normalizar pesos
        weights = weights / weights.sum()
        
        # Crear sampler para balancear clases
        samples_weights = [weights[label] for label in dataset.labels]
        sampler = WeightedRandomSampler(
            weights=samples_weights,
            num_samples=len(dataset),
            replacement=True
        )
        
        # Crear dataloader con el sampler
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=True
        )
        
        # Crear modelo
        logger.info("Inicializando modelo...")
        model = models.resnet34(pretrained=True)
        
        # Congelar capas iniciales
        for param in list(model.parameters())[:-6]:
            param.requires_grad = False
            
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
            nn.Linear(256, 2)  # 2 clases: bueno y malo
        )
        model = model.to(device)
        
        # Criterio con pesos
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        
        # Optimizador con diferentes learning rates
        optimizer = optim.AdamW([
            {'params': model.fc.parameters(), 'lr': learning_rate},
            {'params': model.layer4.parameters(), 'lr': learning_rate * 0.1},
            {'params': model.layer3.parameters(), 'lr': learning_rate * 0.01}
        ], weight_decay=0.01)
        
        # Scheduler mejorado
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.3,
            patience=3,
            verbose=True
        )
        
        # Entrenamiento
        logger.info("Iniciando entrenamiento...")
        best_loss = float('inf')
        best_accuracy = 0.0
        patience = 7  # Aumentado para dataset pequeño
        patience_counter = 0
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Contadores por clase
            class_correct = [0] * 2
            class_total = [0] * 2
            
            for i, (images, labels) in enumerate(dataloader):
                try:
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # Actualizar contadores por clase
                    for j in range(len(labels)):
                        label = labels[j].item()
                        pred = predicted[j].item()
                        class_total[label] += 1
                        if label == pred:
                            class_correct[label] += 1
                    
                except Exception as e:
                    logger.error(f"Error en batch {i}: {str(e)}")
                    continue
            
            epoch_loss = running_loss / len(dataloader)
            accuracy = 100 * correct / total
            
            # Imprimir métricas por clase
            logger.info(f'\nÉpoca [{epoch+1}/{num_epochs}]')
            logger.info(f'Pérdida: {epoch_loss:.4f} - Precisión General: {accuracy:.2f}%')
            
            classes = ['bueno', 'malo']
            for i in range(2):
                class_acc = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
                logger.info(f'Precisión {classes[i]}: {class_acc:.2f}%')
            
            scheduler.step(epoch_loss)
            
            # Guardar mejor modelo basado en precisión
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                logger.info(f"Guardando mejor modelo (precisión: {best_accuracy:.2f}%)")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                    'accuracy': accuracy
                }, 'mejor_modelo.pth')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping después de {patience} épocas sin mejora")
                break
        
        logger.info("Entrenamiento completado!")
        
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        # Verificar estructura de carpetas
        if not os.path.exists('datos_entrenamiento'):
            os.makedirs('datos_entrenamiento/bueno', exist_ok=True)
            os.makedirs('datos_entrenamiento/malo', exist_ok=True)
            logger.info("Carpetas de entrenamiento creadas")
            logger.info("Por favor, coloca las imágenes en las carpetas correspondientes:")
            logger.info("- datos_entrenamiento/bueno/")
            logger.info("- datos_entrenamiento/malo/")
            exit(1)
            
        train_model()
    except Exception as e:
        logger.error(f"Error crítico: {str(e)}")