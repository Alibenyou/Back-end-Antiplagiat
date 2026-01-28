# On utilise l'image standard (pas la slim) qui contient déjà GCC et build-essential
FROM python:3.10

# Définir les variables d'environnement
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# On installe seulement le strict nécessaire pour l'affichage/PDF
# Si apt-get échoue, on continue quand même le build
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 || true

# Gestion des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copier le projet
COPY . .

# Port dynamique pour Render
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]