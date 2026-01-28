import os
import fitz  # PyMuPDF
import requests
import time
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client, ClientOptions
from dotenv import load_dotenv

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
HF_API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def calculate_similarity(text1, text2):
    """ Envoie les textes à Hugging Face avec gestion du temps de chargement """
    try:
        payload = {
            "inputs": {
                "source_sentence": text1,
                "sentences": [text2]
            }
        }
        # On ajoute un Header pour forcer l'attente du modèle
        headers = {"x-use-cache": "false", "x-wait-for-model": "true"}
        
        response = requests.post(HF_API_URL, json=payload, headers=headers, timeout=30)
        result = response.json()

        # Debug: on affiche ce que l'IA répond vraiment dans les logs
        print(f"DEBUG IA response: {result}")

        # Si Hugging Face renvoie une erreur de chargement
        if isinstance(result, dict) and "error" in result:
            print("L'IA charge encore... on attend 10s")
            time.sleep(10)
            return calculate_similarity(text1, text2)

        # Extraction du score (Hugging Face renvoie souvent une liste [0.85])
        if isinstance(result, list) and len(result) > 0:
            return float(result[0])
        
        return 0
    except Exception as e:
        print(f"Erreur IA critique : {e}")
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
        # 1. Récupération du fichier
        file_bin = supabase.storage.from_('fichiers_plagiat').download(file_path)
        user_text = extract_text(file_bin)
        
        if not user_text.strip():
            supabase.table("analyses").update({"status": "termine", "plagiarism_score": 0}).eq("id", analysis_id).execute()
            return

        # 2. Recherche Web
        search_query = user_text[:300] 
        web_sources = search_google(search_query)

        max_score = 0
        for source in web_sources:
            link = source['link']
            web_text = scrape_website(link)

            if web_text:
                # 3. Comparaison via API Hugging Face
                similarity = calculate_similarity(user_text[:1000], web_text[:1000])
                score = round(similarity * 100, 2)
                
                if score > max_score:
                    max_score = score

                if score > 15:
                    supabase.table("analysis_results").insert({
                        "analysis_id": analysis_id,
                        "url": link,
                        "title": source.get('title'),
                        "similarity_score": score
                    }).execute()
        
        # 4. Finalisation
        supabase.table("analyses").update({
            "status": "termine",
            "plagiarism_score": max_score
        }).eq("id", analysis_id).execute()

    except Exception as e:
        print(f"Erreur : {e}")
        supabase.table("analyses").update({"status": "erreur"}).eq("id", analysis_id).execute()

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