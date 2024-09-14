FROM python:3.12.3

# Set the working directory
WORKDIR /main

# Copy the requirements file from 'main'
COPY ./main/requirements.txt /main/requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir --upgrade -r /main/requirements.txt

# Copy the application code from the 'main/app' directory
# COPY ./main/app /main/app
COPY ./main /main

# # Copy the run.py script
# COPY ./main/run.py /main/run.py

# Set environment variables
ENV FLASK_APP=run.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

# Expose port 5000
EXPOSE 5000

# Define the default command (assuming you want to run run.py)
CMD ["flask", "run"]