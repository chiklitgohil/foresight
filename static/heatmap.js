document.addEventListener('DOMContentLoaded', () => {
    const heatmapGrid = document.getElementById('heatmap-grid');
    const statusText = document.getElementById('heatmap-status');
    let nodes = {};

    // Initialize 100 nodes
    for (let i = 0; i < 100; i++) {
        const node = document.createElement('div');
        node.className = 'node node-low';
        node.id = `node-${i}`;
        node.innerText = 'INIT';
        heatmapGrid.appendChild(node);
        nodes[i] = node;
    }

    const fetchFleetData = async () => {
        try {
            statusText.innerText = "Polling fleet data...";
            const response = await fetch('/api/fleet');
            const data = await response.json();
            
            if (data.error) {
                statusText.innerText = "Error fetching data.";
                return;
            }

            // Update nodes
            data.fleet.forEach((machine, i) => {
                if (i >= 100) return;
                const node = nodes[i];
                
                // Set text to short product ID format (e.g. L4718)
                node.innerText = machine.product_id;
                
                // Clear previous classes
                node.className = 'node';
                
                if (machine.risk_level.includes('High')) {
                    node.classList.add('node-high');
                } else if (machine.risk_level.includes('Medium')) {
                    node.classList.add('node-med');
                } else {
                    node.classList.add('node-low');
                }
            });
            
            statusText.innerText = `Last updated: ${new Date().toLocaleTimeString()}`;
        } catch (err) {
            console.error(err);
            statusText.innerText = "Connection lost.";
        }
    };

    // Initial fetch
    fetchFleetData();

    // Poll every 5 seconds
    setInterval(fetchFleetData, 5000);
});
