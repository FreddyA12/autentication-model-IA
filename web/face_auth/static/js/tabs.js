/**
 * Tab Navigation
 */

document.addEventListener('DOMContentLoaded', function() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const faceInfo = document.getElementById('faceInfo');
    const voiceInfo = document.getElementById('voiceInfo');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            const targetTab = this.getAttribute('data-tab');
            
            // Update active button
            tabButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');
            
            // Show corresponding tab content
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            
            const tabContent = document.getElementById(targetTab + 'Tab');
            if (tabContent) {
                tabContent.classList.add('active');
            }
            
            // Update footer info
            const faceInfo = document.getElementById("faceInfo");
            const voiceInfo = document.getElementById("voiceInfo");
            const dualInfo = document.getElementById("dualInfo");

            if (faceInfo)
              faceInfo.style.display = targetTab === "face" ? "block" : "none";
            if (voiceInfo)
              voiceInfo.style.display =
                targetTab === "voice" ? "block" : "none";
            if (dualInfo)
              dualInfo.style.display = targetTab === "dual" ? "block" : "none";
        });
    });
});
