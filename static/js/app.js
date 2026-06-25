// Inisialisasi Canvas
const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');
const canvasHint = document.getElementById('canvasHint');

// State drawing
let isDrawing = false;
let hasDrawn = false;

// Konfigurasi Brush
ctx.strokeStyle = 'black'; // warna hitam
ctx.lineWidth = 20; // ketebalan garis
ctx.lineCap = 'round'; // ujung garis bulat
ctx.lineJoin = 'round'; // sambungan garis bulat

// Helper -> Koordinat mouse pada canvas
function getPos(e){
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    // support touch for mobile & mouse for desktop
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;

    return {
        x: (clientX - rect.left) * scaleX,
        y: (clientY - rect.top) * scaleY,
    };
}

// Event -> Start Drawing
function startDraw(e){
    e.preventDefault();
    isDrawing = true;
    hasDrawn = true;
    const {x, y} = getPos(e);
    ctx.beginPath();
    ctx.moveTo(x, y);

    // Sembunyikan hint canvas saat pertama kali menggambar
    if(!hasDrawn){
        hasDrawn = true;
        canvasHint.classList.add('hidden');
    }
}

// Event -> Drawing
function draw(e){
    if(!isDrawing) return;
    e.preventDefault();
    const {x, y} = getPos(e);
    ctx.lineTo(x, y);
    ctx.stroke();
}

// Event -> Stop Drawing
function stopDraw(e){
    isDrawing = false;
    ctx.beginPath(); // reset path untuk memulai garis baru saat menggambar lagi
}

// Daftarkan semua event listener
canvas.addEventListener('mousedown', startDraw);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDraw);
canvas.addEventListener('mouseleave', stopDraw);

// Support touch events for mobile devices
canvas.addEventListener('touchstart', startDraw, {passive: false});
canvas.addEventListener('touchmove', draw, {passive: false});
canvas.addEventListener('touchend', stopDraw);

// Hapus Canvas
document.getElementById('clearBtn').addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    hasDrawn = false;
    canvasHint.classList.remove('hidden');
    showIdle();
});

// UI State Management
const resultIdle = document.getElementById('resultIdle');
const resultLoading = document.getElementById('resultLoading');
const resultOutput = document.getElementById('resultOutput');

function showIdle(){
    resultIdle.classList.remove('hidden');
    resultLoading.classList.add('hidden');
    resultOutput.classList.add('hidden');
}

function showLoading(){
    resultIdle.classList.add('hidden');
    resultLoading.classList.remove('hidden');
    resultOutput.classList.add('hidden');
}

function showResult(data){
    resultIdle.classList.add('hidden');
    resultLoading.classList.add('hidden');
    resultOutput.classList.remove('hidden');

    // Tampilkan angka yang diprediksi
    document.getElementById('predictedDigit').textContent = data.digit;

    // Tampilkan confidence score
    document.getElementById('confidenceBadge').textContent = data.confidence.toFixed(2) + '%';

    // Render probability bars
    renderProbBars(data.all_probabilities, data.digit);
}

// Render -> Probability Bars
function renderProbBars(probs, topDigit){
    const container = document.getElementById('probBars');
    container.innerHTML = ''; // Clear previous bars

    for (let i = 0; i <= 9; i++){
        const key = String(i);
        const value = probs[key] || 0;
        const isTop = i === topDigit;

        const row = document.createElement('div');
        row.className = 'prob-row';

        row.innerHTML = `
            <span class="prob-label">${i}</span>
            <div class="prob-track">
                <div class="prob-fill ${isTop ? 'is-top' : ''}"
                    style="width: 0%"
                    data-target="${value}">
                </div>
            </div>
            <span class="prob-value">${value.toFixed(1)}%</span>
        `;
        container.appendChild(row);
    }

    // Animasi bars setelah DOM terbentuk
    requestAnimationFrame(() => {
        document.querySelectorAll('.prob-fill').forEach(fill => {
            const target = parseFloat(fill.dataset.target);
            fill.style.width = target + '%';
        });
    });
}

// Tombol Prediksi
document.getElementById('predictBtn').addEventListener('click', async () => {
    // Cek apakah canvas kosong
    if(!hasDrawn){
        alert("Gambar angka lebih dulu di kanvas!");
        return;
    }

    showLoading();

    // Konversi canvas ke base64 PNG
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;
    const tempCtx = tempCanvas.getContext('2d');

    tempCtx.fillStyle = '#ffffff'; // fill putih dulu
    tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
    tempCtx.drawImage(canvas, 0, 0); // overlay gambar asli di atas background putih

    const imageBase64 = tempCanvas.toDataURL('image/png'); // capture dengan bg putih untuk memastikan model bisa mengenali digit dengan benar

    try{
        // Kirim ke FLASK API
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json'},
            body: JSON.stringify({ image: imageBase64 }),
        });

        const data = await response.json();

        if(data.success){
            showResult(data);
        }else {
            alert("Error dari server: "+data.error)
            showIdle();
        }
    } catch (err){
        alert("Tidak bisa terhubung ke sever. Pastikan server Flask berjalan.");
        showIdle();
        console.error(err);
    }
});