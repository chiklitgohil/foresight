document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('telemetry-form');
    const predictBtn = document.getElementById('predict-btn');
    const simBtn = document.getElementById('sim-btn');
    
    const riskText = document.getElementById('risk-text');
    const recText = document.getElementById('rec-text');
    const statusCircle = document.getElementById('status-circle');
    const recBox = document.getElementById('rec-box');
    const latencyDisplay = document.getElementById('latency-display');
    const logBody = document.getElementById('log-body');
    
    let isSimulating = false;
    let simTimeout;
    
    // Web Audio API context for buzzer sounds
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    let audioCtx;

    const playAlertSound = (severity) => {
        if (!audioCtx) audioCtx = new AudioContext();
        if (audioCtx.state === 'suspended') audioCtx.resume();
        
        const oscillator = audioCtx.createOscillator();
        const gainNode = audioCtx.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(audioCtx.destination);
        
        if (severity === 'High') {
            // Harsh buzzer
            oscillator.type = 'square';
            oscillator.frequency.setValueAtTime(150, audioCtx.currentTime);
            oscillator.frequency.setValueAtTime(250, audioCtx.currentTime + 0.1);
            oscillator.frequency.setValueAtTime(150, audioCtx.currentTime + 0.2);
            gainNode.gain.setValueAtTime(0.3, audioCtx.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.8);
            oscillator.start();
            oscillator.stop(audioCtx.currentTime + 1);
        } else if (severity === 'Medium') {
            // Warning chime
            oscillator.type = 'sine';
            oscillator.frequency.setValueAtTime(440, audioCtx.currentTime); // A4
            gainNode.gain.setValueAtTime(0.2, audioCtx.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.5);
            oscillator.start();
            oscillator.stop(audioCtx.currentTime + 0.5);
        }
    };

    const logIncident = (risk_level, recommendation) => {
        const product_id = document.getElementById('product_id').value;
        const true_mode = document.getElementById('true_failure_mode').value;
        const time = new Date().toLocaleTimeString();
        
        const row = document.createElement('tr');
        
        let rowClass = "";
        let severityText = "Low";
        
        if (risk_level.includes('High')) {
            rowClass = "log-high";
            severityText = "CRITICAL";
        } else if (risk_level.includes('Medium')) {
            rowClass = "log-med";
            severityText = "WARNING";
        }
        
        row.className = rowClass;
        row.innerHTML = `
            <td>${time}</td>
            <td><strong>${product_id}</strong></td>
            <td>${severityText}</td>
            <td>${true_mode !== 'None' ? true_mode : 'Unknown'}</td>
            <td>${recommendation}</td>
        `;
        
        logBody.prepend(row);
        
        // Keep only last 10 logs
        if (logBody.children.length > 10) {
            logBody.lastChild.remove();
        }
    };

    // Reset visual classes
    const resetClasses = () => {
        statusCircle.className = 'status-circle';
        riskText.className = '';
        recBox.className = 'recommendation-box';
        document.body.className = '';
    };

    const updateUI = (risk_level, recommendation, latency) => {
        resetClasses();
        
        riskText.innerText = risk_level;
        recText.innerText = recommendation || "No action required.";
        latencyDisplay.innerText = `Latency: ${latency.toFixed(2)} ms`;

        let delayNext = 1500; // default simulation delay

        if (risk_level.includes('High')) {
            statusCircle.classList.add('status-high');
            riskText.classList.add('text-high');
            recBox.classList.add('border-high');
            document.body.classList.add('bg-alert-high');
            playAlertSound('High');
            logIncident(risk_level, recommendation);
            delayNext = 4000; // Wait longer so judges can read it
        } else if (risk_level.includes('Medium')) {
            statusCircle.classList.add('status-med');
            riskText.classList.add('text-med');
            recBox.classList.add('border-med');
            document.body.classList.add('bg-alert-med');
            playAlertSound('Medium');
            logIncident(risk_level, recommendation);
            delayNext = 4000; // Wait longer
        } else {
            statusCircle.classList.add('status-low');
            riskText.classList.add('text-low');
            recBox.classList.add('border-low');
        }
        
        return delayNext;
    };

    const runAnalysis = async () => {
        const payload = {
            machine_type: document.getElementById('machine_type').value,
            air_temp: parseFloat(document.getElementById('air_temp').value),
            process_temp: parseFloat(document.getElementById('process_temp').value),
            rpm: parseFloat(document.getElementById('rpm').value),
            torque: parseFloat(document.getElementById('torque').value),
            tool_wear: parseFloat(document.getElementById('tool_wear').value)
        };

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await response.json();
            
            if (data.error) {
                console.error("API Error:", data.error);
                riskText.innerText = "ERROR";
                return 1500;
            }
            
            return updateUI(data.risk_level, data.recommendation, data.latency_ms);
        } catch (err) {
            console.error("Fetch Error:", err);
            riskText.innerText = "OFFLINE";
            return 1500;
        }
    };

    predictBtn.addEventListener('click', () => {
        runAnalysis();
    });

    // Simulation Loop using real dataset streaming
    const simulateStep = async () => {
        if (!isSimulating) return;

        try {
            // 1. Fetch real row from dataset
            const streamRes = await fetch('/api/stream');
            const rowData = await streamRes.json();
            
            if (!rowData.error) {
                // Populate UI
                document.getElementById('product_id').value = rowData.product_id;
                document.getElementById('machine_type').value = rowData.machine_type;
                document.getElementById('air_temp').value = rowData.air_temp;
                document.getElementById('process_temp').value = rowData.process_temp;
                document.getElementById('rpm').value = rowData.rpm;
                document.getElementById('torque').value = rowData.torque;
                document.getElementById('tool_wear').value = rowData.tool_wear;
                document.getElementById('true_failure_mode').value = rowData.true_failure_mode;
            }
        } catch (e) {
            console.error("Simulation streaming error:", e);
        }

        // 2. Predict on the newly populated UI data
        const delayNext = await runAnalysis();
        
        // 3. Queue next step
        if (isSimulating) {
            simTimeout = setTimeout(simulateStep, delayNext);
        }
    };

    simBtn.addEventListener('click', () => {
        isSimulating = !isSimulating;
        if (isSimulating) {
            simBtn.innerText = "Stop Simulation";
            simBtn.style.color = "#ef4444";
            simBtn.style.borderColor = "#ef4444";
            simulateStep();
        } else {
            simBtn.innerText = "Start Simulation";
            simBtn.style.color = "";
            simBtn.style.borderColor = "";
            clearTimeout(simTimeout);
        }
    });
});
