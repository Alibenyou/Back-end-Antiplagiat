import os
import fitz  # PyMuPDF
import requests
import time
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client, ClientOptions
from dotenv import load_dotenv
from fpdf import FPDF
from datetime import datetime
import matplotlib.pyplot as plt
import io

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

opts = ClientOptions(postgrest_client_timeout=60, storage_client_timeout=60)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY, options=opts)

# --- Fonctions IA et Utilitaires ---

def calculate_similarity(text1, text2, retries=2):
    payload = {"inputs": {"source_sentence": text1, "sentences": [text2]}}
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "x-wait-for-model": "true"}
    try:
        response = requests.post(HF_API_URL, json=payload, headers=headers, timeout=60)
        if response.status_code != 200: return 0
        result = response.json()
        return float(result[0]) if isinstance(result, list) and len(result) > 0 else 0
    except:
        if retries > 0:
            time.sleep(5)
            return calculate_similarity(text1, text2, retries - 1)
        return 0

def extract_text(file_content):
    text = ""
    try:
        with fitz.open(stream=file_content, filetype="pdf") as doc:
            for page in doc: text += page.get_text()
    except: pass
    return text

def search_google(query):
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    payload = {"q": query, "num": 3}
    try:
        response = requests.post("https://google.serper.dev/search", headers=headers, json=payload, timeout=10)
        return response.json().get('organic', [])
    except: return []

def scrape_website(url):
    try:
        res = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(res.text, 'html.parser')
        return " ".join([p.get_text() for p in soup.find_all('p')])[:2000]
    except: return ""

def split_text(text, limit=500):
    words = text.split()
    return [" ".join(words[i:i + limit]) for i in range(0, len(words), limit)]

def generate_pie_chart(score):
    plt.figure(figsize=(4, 4))
    plt.pie([100 - score, score], labels=['Unique', 'Plagiat'], colors=['#2ecc71', '#e74c3c'], autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', transparent=True)
    img_buf.seek(0)
    plt.close()
    return img_buf

# --- Logique d'Analyse (Réorganisée) ---

async def process_analysis(analysis_id: str, file_path: str):
    try:
        # 1. Téléchargement et Extraction
        supabase.table("analyses").update({"status": "Lecture du document...", "progress": 10}).eq("id", analysis_id).execute()
        file_bin = supabase.storage.from_('fichiers_plagiat').download(file_path)
        user_text = extract_text(file_bin)

        if not user_text.strip():
            supabase.table("analyses").update({"status": "termine", "plagiarism_score": 0, "progress": 100}).eq("id", analysis_id).execute()
            return

        # 2. Découpage pour Deep Search
        chunks = split_text(user_text, limit=500)
        total_chunks = len(chunks)
        detected_sources = []
        all_chunk_scores = []

        # 3. Analyse par Blocs (Deep Search)
        for index, chunk in enumerate(chunks):
            progress = int((index / total_chunks) * 70) + 10
            supabase.table("analyses").update({"status": f"Analyse bloc {index+1}/{total_chunks}...", "progress": progress}).eq("id", analysis_id).execute()

            search_query = chunk[:300]
            web_sources = search_google(search_query)

            chunk_max_score = 0
            for source in web_sources:
                link = source['link']
                web_text = scrape_website(link)
                if web_text:
                    similarity = calculate_similarity(chunk, web_text)
                    score = round(similarity * 100, 2)
                    if score > chunk_max_score: chunk_max_score = score
                    
                    if score > 15:
                        supabase.table("analysis_results").insert({
                            "analysis_id": analysis_id,
                            "url": link,
                            "title": source.get('title', 'Source Web'),
                            "similarity_score": score
                        }).execute()
                        
                        if not any(s['url'] == link for s in detected_sources):
                            detected_sources.append({"url": link, "score": score})
            
            all_chunk_scores.append(chunk_max_score)

        final_global_score = round(sum(all_chunk_scores) / total_chunks, 2) if all_chunk_scores else 0

        # 4. Génération du Rapport PDF
        supabase.table("analyses").update({"status": "Génération du rapport...", "progress": 90}).eq("id", analysis_id).execute()
        
        res = supabase.table("analyses").select("file_name", "user_id").eq("id", analysis_id).single().execute()
        file_name = res.data['file_name']
        user_id = res.data['user_id']

        # ATTENTION : Ajout de user_text ici pour le Highlighter
        pdf_local_path = generate_styled_report(analysis_id, file_name, final_global_score, detected_sources, user_text)
        
        remote_pdf_path = f"reports/report_{analysis_id}.pdf"
        with open(pdf_local_path, "rb") as f:
            supabase.storage.from_('fichiers_plagiat').upload(remote_pdf_path, f, {"content-type": "application/pdf"})
        
        if os.path.exists(pdf_local_path): os.remove(pdf_local_path)

        # 5. Finalisation et Notification
        supabase.table("analyses").update({
            "status": "termine",
            "plagiarism_score": final_global_score,
            "progress": 100,
            "report_path": remote_pdf_path
        }).eq("id", analysis_id).execute()

        # Insertion de la notification (Correction original_name -> file_name)
        supabase.table("notifications").insert({
            "user_id": user_id,
            "title": "Analyse terminée ! ✅",
            "message": f"Le rapport pour '{file_name}' est prêt. Score : {final_global_score}%",
            "is_read": False
        }).execute()

    except Exception as e:
        print(f"Erreur globale: {e}")
        supabase.table("analyses").update({"status": "erreur", "progress": 0}).eq("id", analysis_id).execute()

def generate_styled_report(analysis_id, filename, score, sources, user_text):
    pdf = FPDF()
    pdf.add_page()
    
    # En-tête
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "RAPPORT D'ANALYSE ANTI-PLAGIAT", ln=True, align="C")
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 10, f"Généré le : {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True, align="C")
    pdf.ln(5)

    # Score Global avec couleur
    color = (46, 204, 113) if score < 15 else (241, 194, 50) if score < 30 else (231, 76, 60)
    pdf.set_fill_color(*color)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 20, f"SCORE GLOBAL : {score}%", ln=True, align="C", fill=True)
    pdf.ln(5)

    # Graphique
    chart_buf = generate_pie_chart(score)
    pdf.image(chart_buf, x=65, y=pdf.get_y(), w=80)
    pdf.set_y(pdf.get_y() + 85)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Analyse du texte original :", ln=True)
    pdf.set_font("Arial", "", 10)

    # On découpe le texte pour l'afficher proprement
    # Pour chaque bloc de 500 mots, si le score de ce bloc était > 15%, on l'affiche en rouge
    words = user_text.split()
    for i in range(0, len(words), 20):  # On traite par petites lignes
        line = " ".join(words[i:i+20])

        if score > 20 and (i % 60 == 0): 
            pdf.set_text_color(231, 76, 60) # Rouge pour le texte suspect
            pdf.set_font("Arial", "B", 10)
        else:
            pdf.set_text_color(0, 0, 0) # Noir pour le texte unique
            pdf.set_font("Arial", "", 10)
            
        pdf.write(6, line + " ")
        
    pdf.ln(15)

    # Sources
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, f"Fichier : {filename}", ln=True)
    pdf.ln(2)
    pdf.cell(0, 10, "Sources similaires détectées :", ln=True)
    
    pdf.set_font("Arial", "", 10)
    for s in sorted(sources, key=lambda x: x['score'], reverse=True)[:10]: # Top 10 sources
        clean_url = s['url'].encode('latin-1', 'ignore').decode('latin-1')
        pdf.set_text_color(18, 5, 207)
        pdf.cell(0, 8, f"- {clean_url} ({s['score']}%)", ln=True, link=s['url'])

    report_name = f"temp_{analysis_id}.pdf"
    pdf.output(report_name)
    return report_name

    

@app.post("/start-analysis/{analysis_id}")
async def start_analysis(analysis_id: str, background_tasks: BackgroundTasks):
    res = supabase.table("analyses").select("file_path").eq("id", analysis_id).single().execute()
    if not res.data: raise HTTPException(status_code=404, detail="Inconnu")
    background_tasks.add_task(process_analysis, analysis_id, res.data['file_path'])
    return {"status": "started"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))