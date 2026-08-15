// In script.js, look for this part:
fetch('/expand', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({query: query})
})
.then(response => response.json())
.then(data => {
    // Display ALL expanded terms
    const terms = data.expanded_terms || [];
    const termsList = document.getElementById('terms-list');
    
    // Clear previous results
    termsList.innerHTML = '';
    
    // Show ALL terms (not just the first one)
    terms.forEach(term => {
        const li = document.createElement('li');
        li.textContent = term;
        termsList.appendChild(li);
    });
    
    // Update metrics
    document.getElementById('precision').textContent = data.metrics.precision.toFixed(2);
    document.getElementById('recall').textContent = data.metrics.recall.toFixed(2);
    document.getElementById('f1').textContent = data.metrics.f1.toFixed(2);
})
.catch(error => {
    console.error('Error:', error);
});