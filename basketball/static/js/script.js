document.addEventListener("DOMContentLoaded", function() {
    const swiper = new Swiper('.swiper-container', {
        loop: true,
        autoplay: {
            delay: 5000,
        },
        pagination: {
            el: '.swiper-pagination',
            clickable: true,
        },
    });

    // Schedule Table Data
    const games = [
        { date: "2024-10-01", team1: "Team A", team2: "Team B", result: "80-76" },
        { date: "2024-10-02", team1: "Team C", team2: "Team D", result: "65-70" }
    ];

    const table = document.querySelector('#schedule table');
    
    if (table) {
        const rows = games.map(game => {
            const row = document.createElement('tr');
            row.innerHTML = `<td>${game.date}</td><td>${game.team1}</td><td>${game.team2}</td><td>${game.result}</td>`;
            return row;
        });

        const tableBody = document.createDocumentFragment();
        rows.forEach(row => tableBody.appendChild(row));
        table.appendChild(tableBody);
    } else {
        console.error("Table not found in the document.");
    }
    
});
