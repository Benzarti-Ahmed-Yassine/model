# Utiliser l'image Python officielle minimale
FROM python:3.9-slim

# Créer le répertoire /app avant de changer les permissions
RUN mkdir -p /app && useradd -m appuser && chown appuser:appuser /app

# Utiliser un utilisateur non-root
USER root

# Définir le répertoire de travail
WORKDIR /app

# Copier et installer les dépendances
COPY --chown=appuser:appuser requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY --chown=appuser:appuser . .

# Vérification de l'état pour la surveillance
HEALTHCHECK --interval=30s --timeout=3s CMD curl -f http://localhost:5001/health || exit 1

# Commande de démarrage de l'application Flask
CMD ["python", "app.py"]

