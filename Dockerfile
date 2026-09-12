FROM nikolaik/python-nodejs:python3.11-nodejs20
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
CMD ["bash", "start.sh"]
