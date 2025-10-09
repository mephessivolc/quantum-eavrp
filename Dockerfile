
FROM jupyter/base-notebook:latest 

WORKDIR /home/jovyan/work

COPY . /home/jovyan/work

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 4000

# CMD ["bash"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=4000", "--allow-root"]