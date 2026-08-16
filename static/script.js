document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("query-form");
    const input = document.getElementById("query");
    const resultContainer = document.getElementById("results");
    const loader = document.getElementById("loader");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const query = input.value.trim();
        if (!query) return;

        loader.style.display = "block";
        resultContainer.innerHTML = "";

        try {
            const response = await fetch("/expand", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({ query })
            });

            if (!response.ok) throw new Error(await response.text());
            
            const data = await response.json();
            const terms = data.expanded_terms || [];
            const metrics = data.metrics || {};
            
            resultContainer.innerHTML = `
                <h3>🔍 Original Query (${data.detected_domain || 'general'}):</h3>
                <p>${query}</p>
                
                <h3>📌 Expanded Terms:</h3>
                <ul id="expanded-terms">
                    ${terms.map(term => `
                        <li>
                            <a href="https://en.wikipedia.org/wiki/${encodeURIComponent(term)}"
                               target="_blank"
                               class="expanded-term-link">${term}</a>
                        </li>`).join("")}
                </ul>
                
                <h3>📊 Evaluation Metrics:</h3>
                <div class="metric-boxes">
                    <div class="box"><strong>Precision</strong><br>${metrics.precision?.toFixed(2) || 0}</div>
                    <div class="box"><strong>Recall</strong><br>${metrics.recall?.toFixed(2) || 0}</div>
                    <div class="box"><strong>F1 Score</strong><br>${metrics.f1?.toFixed(2) || 0}</div>
                    <div class="box"><strong>Perplexity</strong><br>${data.perplexity?.toFixed(2) || 'N/A'}</div>
                </div>
            `;

        } catch (err) {
            resultContainer.innerHTML = `<p class="error">⚠️ Error: ${err.message}</p>`;
        } finally {
            loader.style.display = "none";
        }
    });
});
