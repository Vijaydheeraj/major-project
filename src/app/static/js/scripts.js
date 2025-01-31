function startAnalysis() {
    fetch('/start-analysis', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        console.log(data.message);
        alert(data.message);
    })
    .catch(error => {
        console.error('Erreur:', error);
        alert("Erreur lors du lancement de l'analyse.");
    });
}