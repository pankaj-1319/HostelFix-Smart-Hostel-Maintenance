document.querySelectorAll('.flash').forEach(el => setTimeout(()=>el.remove(),3500));
function filterTable(){
 const q=(document.getElementById('search')?.value||'').toLowerCase();
 document.querySelectorAll('#complaintTable tbody tr').forEach(r=>r.style.display=r.innerText.toLowerCase().includes(q)?'':'none');
}
