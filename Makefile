
devenv:
	@docker run -it -p 8080:8080 -v "${PWD}/app:/app" -v "${PWD}/data:/data" registry.gitlab.com/jeff27101/esun-ai-competition /bin/bash

build:
	@docker build --platform linux/amd64 -t registry.gitlab.com/jeff27101/esun-ai-competition .

run:
	@docker run -it -p 8080:8080 -v "${PWD}/app:/app" -v "${PWD}/data:/data" registry.gitlab.com/jeff27101/esun-ai-competition

push:
	@docker push registry.gitlab.com/jeff27101/esun-ai-competition
