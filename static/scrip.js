document.addEventListener('DOMContentLoaded', function() {
    const imageInput = document.getElementById('imageInput');
    const imagePreview = document.getElementById('imagePreview');
    const resultado = document.getElementById('resultado');
    const loadingSpinner = document.getElementById('loadingSpinner');

    if (imageInput) {
        imageInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                // Mostrar vista previa
                const reader = new FileReader();
                reader.onload = function(e) {
                    imagePreview.src = e.target.result;
                    imagePreview.style.display = 'block';
                }
                reader.readAsDataURL(file);

                // Enviar imagen para análisis
                const formData = new FormData();
                formData.append('file', file);

                // Mostrar spinner y ocultar resultado anterior
                loadingSpinner.style.display = 'block';
                resultado.style.display = 'none';

                fetch('/predict', {  // Cambiado de /upload a /predict
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    loadingSpinner.style.display = 'none';
                    resultado.style.display = 'block';

                    if (data.error) {
                        resultado.innerHTML = `
                            <div class="resultado-error">
                                <i class="fas fa-exclamation-circle"></i>
                                <p>Error: ${data.error}</p>
                            </div>
                        `;
                    } else {
                        const estiloResultado = data.prediccion === 'bueno' ? 'resultado-bueno' : 'resultado-malo';
                        const iconoResultado = data.prediccion === 'bueno' ? 'fa-check-circle' : 'fa-times-circle';
                        
                        resultado.innerHTML = `
                            <div class="${estiloResultado}">
                                <i class="fas ${iconoResultado}"></i>
                                <h3>Resultado: ${data.prediccion.toUpperCase()}</h3>
                                <div class="probabilidades">
                                    <p><i class="fas fa-thumbs-up"></i> Probabilidad bueno: ${data.probabilidad_bueno}</p>
                                    <p><i class="fas fa-thumbs-down"></i> Probabilidad malo: ${data.probabilidad_malo}</p>
                                </div>
                            </div>
                        `;
                    }
                })
                .catch(error => {
                    loadingSpinner.style.display = 'none';
                    resultado.style.display = 'block';
                    resultado.innerHTML = `
                        <div class="resultado-error">
                            <i class="fas fa-exclamation-circle"></i>
                            <p>Error al procesar la imagen</p>
                        </div>
                    `;
                    console.error('Error:', error);
                });
            }
        });
    }
});