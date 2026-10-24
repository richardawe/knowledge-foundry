# The research service: `foundry serve-research` behind contract foundry.research/v1.
#
#   docker build -t knowledge-foundry .
#   docker run -p 8100:8100 -e FOUNDRY_API_TOKEN=... -e FOUNDRY_CONTACT_EMAIL=... \
#              -v foundry-data:/data knowledge-foundry
#
# No model is needed. FOUNDRY_VERIFY_PROVIDER/MODEL (and that provider's key)
# add a model check to corroboration; without them the sentence rule stands alone.
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY templates ./templates
RUN pip install --no-cache-dir ".[research]"

# Company knowledge areas and their indexes hold third-party findings: keep them
# on a volume, never in the image.
ENV FOUNDRY_ROOT=/app FOUNDRY_VAR=/data/var FOUNDRY_INSTANCES=/data/companies
RUN mkdir -p /data && chown nobody /data
VOLUME /data
EXPOSE 8100
USER nobody
CMD ["foundry", "serve-research", "--host", "0.0.0.0", "--port", "8100"]
