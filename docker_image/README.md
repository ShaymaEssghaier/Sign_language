---
title: Sign Language
emoji: ⚡
colorFrom: gray
colorTo: pink
sdk: docker
pinned: false
license: mit
---

# Sign Language Application

## Hugging Face Spaces Repository

This Sign Language recognition application is deployed on Hugging Face Spaces using Docker:

- **Hugging Face Spaces Repository**: [project-sign-language/Sign_language](https://huggingface.co/spaces/project-sign-language/Sign_language)

## GitHub Repository

The source code for this application is also available on GitHub:

- **GitHub Repository**: [ShaymaEssghaier/Sign_language](https://github.com/ShaymaEssghaier/Sign_language)

## Docker Repository

The Docker image used to deploy this application is also available on Docker Hub:

- **Docker Repository**: [remifigea/sign_language](https://hub.docker.com/r/remifigea/sign_language)

## Dockerfiles

### Dockerfile

Use this Dockerfile to run the application in local development mode.

To build and run the image locally:
```bash
docker build -f Dockerfile -t my-app:local .
docker run -p 5000:5000 my-app:local
```

### Dockerfile.production

Use this Dockerfile to run the application in production mode. This is the one used on the Hugging Face Hub repository after renaming it to Dockerfile.



