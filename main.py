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

opts = ClientOptions(postgrest_client_timeout=60, storage_client_timeout=60)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY, options=opts)

# --- Configuration IA via API ---
# On utilise le même modèle mais hébergé chez Hugging Face

import os

# On récupère le token depuis les variables d'environnement de Render
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def calculate_similarity(text1, text2, retries=2):
    payload = {
        "inputs": {
            "source_sentence": text1,
            "sentences": [text2]
        }
    }
    
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "x-wait-for-model": "true" # Indispensable pour réveiller le modèle
    }
    
    try:
        # On passe le timeout à 60 secondes pour laisser le temps à l'IA de charger
        response = requests.post(HF_API_URL, json=payload, headers=headers, timeout=60)
        
        if response.status_code != 200:
            print(f"Erreur Router ({response.status_code}): {response.text}")
            return 0
            
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return float(result[0])
        return 0

    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        if retries > 0:
            print(f"L'IA est lente à répondre, nouvel essai... ({retries} restants)")
            time.sleep(5)
            return calculate_similarity(text1, text2, retries - 1)
        return 0
    except Exception as e:
        print(f"Erreur technique : {e}")
        return 0

# --- Fonctions Utilitaires ---

def extract_text(file_content):
    text = ""
    try:
        with fitz.open(stream=file_content, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Erreur PDF : {e}")
    return text

def search_google(query):
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    payload = {"q": query, "num": 3} # Limité à 3 pour la rapidité
    try:
        response = requests.post("https://google.serper.dev/search", headers=headers, json=payload, timeout=10)
        return response.json().get('organic', [])
    except:
        return []

def scrape_website(url):
    try:
        res = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(res.text, 'html.parser')
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        return " ".join(paragraphs)[:2000] # On limite le texte récupéré
    except:
        return ""

# --- Logique d'Analyse ---

async def process_analysis(analysis_id: str, file_path: str):
    try:
        # --- Étape 1 : Lecture (20%) ---
        supabase.table("analyses").update({"status": "Lecture du document...", "progress": 20}).eq("id", analysis_id).execute()
        
        file_bin = supabase.storage.from_('fichiers_plagiat').download(file_path)
        user_text = extract_text(file_bin)
        
        if not user_text.strip():
            supabase.table("analyses").update({"status": "termine", "plagiarism_score": 0, "progress": 100}).eq("id", analysis_id).execute()
            return

        # --- Étape 2 : Recherche Web (50%) ---
        supabase.table("analyses").update({"status": "Recherche de sources web...", "progress": 50}).eq("id", analysis_id).execute()
        
        search_query = user_text[:300] 
        web_sources = search_google(search_query)

        max_score = 0
        detected_sources = [] # Pour le rapport PDF

        # --- Étape 3 : Comparaison IA (80%) ---
        supabase.table("analyses").update({"status": "Analyse de similitude IA...", "progress": 80}).eq("id", analysis_id).execute()
        
        for source in web_sources:
            link = source['link']
            web_text = scrape_website(link)

            if web_text:
                similarity = calculate_similarity(user_text[:1000], web_text[:1000])
                score = round(similarity * 100, 2)
                
                if score > max_score:
                    max_score = score

                if score > 15:
                    source_data = {
                        "analysis_id": analysis_id,
                        "url": link,
                        "title": source.get('title'),
                        "similarity_score": score
                    }
                    supabase.table("analysis_results").insert(source_data).execute()
                    detected_sources.append({"url": link, "score": score})
        
        # --- Étape 4 : Génération du Rapport PDF ---
        supabase.table("analyses").update({"status": "Génération du rapport PDF...", "progress": 90}).eq("id", analysis_id).execute()
        
        # On récupère le nom du fichier pour le PDF
        res = supabase.table("analyses").select("file_name").eq("id", analysis_id).single().execute()
        original_name = res.data['file_name']
        
        # Génération locale
        pdf_local_path = generate_styled_report(analysis_id, original_name, max_score, detected_sources)
        
        # Upload du PDF vers Supabase Storage
        remote_pdf_path = f"reports/report_{analysis_id}.pdf"
        with open(pdf_local_path, "rb") as f:
            supabase.storage.from_('fichiers_plagiat').upload(remote_pdf_path, f, {"content-type": "application/pdf"})
        
        # Suppression du fichier local temporaire
        if os.path.exists(pdf_local_path):
            os.remove(pdf_local_path)

        # --- Étape 5 : Finalisation (100%) ---
        supabase.table("analyses").update({
            "status": "termine",
            "plagiarism_score": max_score,
            "progress": 100,
            "report_path": remote_pdf_path # On stocke le chemin du PDF
        }).eq("id", analysis_id).execute()

    except Exception as e:
        print(f"Erreur : {e}")
        supabase.table("analyses").update({"status": "erreur", "progress": 0}).eq("id", analysis_id).execute()


def generate_styled_report(analysis_id, filename, score, sources):
    pdf = FPDF()
    pdf.add_page()
    
    # --- EN-TÊTE ---
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "RAPPORT D'ANALYSE ANTI-PLAGIAT", ln=True, align="C")
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 10, f"Généré le : {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True, align="C")
    pdf.ln(10)

    # --- JAUGE DE SCORE ---
    # Définition de la couleur (Vert, Orange, Rouge)
    if score < 10: color = (46, 204, 113)  # Vert
    elif score < 25: color = (241, 194, 50) # Orange
    else: color = (231, 76, 60)             # Rouge

    pdf.set_fill_color(*color)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 24)
    pdf.cell(0, 25, f"SCORE GLOBAL : {score}%", ln=True, align="C", fill=True)
    pdf.ln(10)

    # --- DÉTAILS DU FICHIER ---
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, f"Fichier analysé : {filename}", ln=True)
    pdf.ln(5)

    # --- TABLEAU DES SOURCES ---
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Sources similaires détectées :", ln=True)
    pdf.set_font("Arial", "", 10)
    
    for source in sources:
        # source pourrait être {"url": "...", "score": 85}
        url = source.get('url', 'Source inconnue')
        sim = source.get('score', 0)
        pdf.multi_cell(0, 8, f"• {url} (Similitude : {sim}%)", border=0)
        pdf.ln(2)

    # --- PIED DE PAGE ---
    pdf.set_y(-30)
    pdf.set_font("Arial", "I", 8)
    pdf.cell(0, 10, "Ce rapport a été généré automatiquement par Antiplagiat IA.", align="C")

    # Sauvegarde locale temporaire avant upload vers Supabase Storage
    report_name = f"report_{analysis_id}.pdf"
    pdf.output(report_name)
    return report_name

@app.post("/start-analysis/{analysis_id}")
async def start_analysis(analysis_id: str, background_tasks: BackgroundTasks):
    res = supabase.table("analyses").select("file_path").eq("id", analysis_id).single().execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Analyse non trouvée")
    
    background_tasks.add_task(process_analysis, analysis_id, res.data['file_path'])
    return {"message": "Analyse lancée"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)