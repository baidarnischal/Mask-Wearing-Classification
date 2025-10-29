FROM python:3.11-slim 
WORKDIR /app

# Install dependencies
COPY api/requirements.txt /app/api/requirements.txt
RUN pip install -r /app/api/requirements.txt

# Copy only necessary files
COPY api /app/api
COPY frontend /app/frontend
COPY models/3_hypertuned_250_epochs /app/models/3_hypertuned_250_epochs


EXPOSE 8000
CMD [ "python", "api/main.py" ]

#only changed files will be built again whenerver 
#TO DO
# multistage building
# ignoring the 2 models and only restricting the 3rd one --> in git --> use ! aagadi
