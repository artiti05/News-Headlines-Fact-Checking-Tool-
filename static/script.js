document.addEventListener('DOMContentLoaded', () => {
    const verifyBtn = document.getElementById('verifyBtn');
    const newsTitle = document.getElementById('newsTitle');
    const newsText = document.getElementById('newsText');
    const statusSteps = document.getElementById('statusSteps');
    const resultsContainer = document.getElementById('resultsContainer');
    const loader = document.getElementById('loader');
    const btnText = verifyBtn.querySelector('.btn-text');
    const historyList = document.getElementById('historyList');

    // Load history on start
    loadHistory();

    verifyBtn.addEventListener('click', async () => {
        const title = newsTitle.value.trim();
        const text = newsText.value.trim();

        if (!text) {
            alert('يرجى إدخال نص الخبر');
            return;
        }

        // Reset UI
        resultsContainer.style.display = 'none';
        statusSteps.style.display = 'flex';
        resetSteps();
        
        // Disable button
        verifyBtn.disabled = true;
        loader.style.display = 'block';
        btnText.style.opacity = '0';

        try {
            // Start process
            setActiveStep(1);
            
            const response = await fetch('/verify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title, text })
            });

            const data = await response.json();

            if (data.error) {
                alert('خطأ: ' + data.error);
                return;
            }

            // Progress visualization
            await completeStep(1);
            setActiveStep(2);
            await sleep(500);
            await completeStep(2);
            setActiveStep(3);
            await sleep(500);
            await completeStep(3);

            // Display results
            displayResults(data);
            loadHistory();

        } catch (error) {
            console.error('Error:', error);
            alert('حدث خطأ أثناء الاتصال بالخادم');
        } finally {
            verifyBtn.disabled = false;
            loader.style.display = 'none';
            btnText.style.opacity = '1';
        }
    });

    function resetSteps() {
        document.querySelectorAll('.step').forEach(s => {
            s.classList.remove('active', 'completed');
        });
    }

    function setActiveStep(n) {
        document.getElementById(`step${n}`).classList.add('active');
    }

    async function completeStep(n) {
        document.getElementById(`step${n}`).classList.remove('active');
        document.getElementById(`step${n}`).classList.add('completed');
    }

    function sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    function displayResults(data) {
        resultsContainer.style.display = 'block';
        document.getElementById('extractedClaim').textContent = data.claim;
        document.getElementById('aiReasoning').textContent = data.reasoning || "لا يوجد تفسير متاح.";
        
        const badge = document.getElementById('verdictBadge');
        const arVerdict = data.ar_verdict || translateVerdict(data.verdict);
        
        badge.textContent = arVerdict;
        badge.className = 'verdict-badge ' + getVerdictClass(data.verdict);

        const list = document.getElementById('evidenceList');
        list.innerHTML = '';
        
        // Show aggregate trust indicator
        const trustDiv = document.createElement('div');
        trustDiv.className = 'trust-indicator';
        const trustLabel = translateTrust(data.source_trust);
        trustDiv.innerHTML = `موثوقية المصادر: <span class="cred-badge cred-${data.source_trust.toLowerCase()}">${trustLabel}</span>`;
        list.appendChild(trustDiv);

        data.evidence.forEach(item => {
            const div = document.createElement('div');
            div.className = 'evidence-item';
            div.innerHTML = `
                <div class="evidence-source">
                    <span>المصدر: <a href="${item.url}" target="_blank">${item.source}</a></span>
                </div>
                <p>${item.snippet}</p>
            `;
            list.appendChild(div);
        });

        // Scroll to results
        resultsContainer.scrollIntoView({ behavior: 'smooth' });
    }

    function translateVerdict(v) {
        if (v === 'TRUE' || v === 'True') return 'صحيح ✅';
        if (v === 'FALSE' || v === 'False') return 'خاطئ ❌';
        return 'غير مؤكد ⚠️';
    }

    function translateTrust(t) {
        if (t === 'High') return 'عالية';
        if (t === 'Medium') return 'متوسطة';
        return 'منخفضة';
    }

    function translateCred(c) {
        if (c === 'credible' || c === 'true') return 'موثوق';
        if (c === 'not credible' || c === 'false') return 'غير موثوق';
        return 'غير معروف';
    }

    function getVerdictClass(v) {
        if (v === 'TRUE' || v === 'True') return 'verdict-true';
        if (v === 'FALSE' || v === 'False') return 'verdict-false';
        return 'verdict-neutral';
    }

    async function loadHistory() {
        try {
            const res = await fetch('/history');
            const data = await res.json();
            historyList.innerHTML = '';
            
            data.forEach(item => {
                const div = document.createElement('div');
                div.className = 'history-item';
                div.innerHTML = `
                    <div class="history-title">${item.title}</div>
                    <div class="history-meta">
                        <span class="verdict-text ${getVerdictClass(item.verdict)}">${item.verdict}</span>
                        <span>${item.timestamp.split(' ')[0]}</span>
                    </div>
                `;
                historyList.appendChild(div);
            });
        } catch (e) {
            console.error('History load error', e);
        }
    }
});

