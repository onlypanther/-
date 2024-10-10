function uploadImage() {
    const imageInput = document.getElementById('file-upload'); // Obtiene el input de archivo
    const file = imageInput.files[0]; // Obtiene el primer archivo seleccionado

    if (file) {
        const formData = new FormData(); // Crea un nuevo objeto FormData
        formData.append('file', file); // Agrega el archivo al FormData

        // Envía la imagen al servidor con Fetch API
        fetch('/upload', {
            method: 'POST',
            body: formData // El cuerpo de la solicitud es el FormData
        })
        .then(response => response.json()) // Convierte la respuesta a JSON
        .then(data => {
            const messageElement = document.getElementById('responseMessage'); // Elemento para mostrar mensajes
            if (data.message) {
                messageElement.innerHTML = `Éxito: ${data.message}`; // Muestra mensaje de éxito
            } else if (data.error) {
                messageElement.innerHTML = `Error: ${data.error}`; // Muestra mensaje de error
            }
        })
        .catch(error => {
            alert('Hubo un error al subir la imagen.'); // Manejo de errores
        });
    } else {
        alert('Por favor, selecciona una imagen.'); // Mensaje si no se selecciona un archivo
    }
}
