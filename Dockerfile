FROM public.ecr.aws/lambda/python:3.12

# Copy requirements.txt
#COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install the specified packages
#RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
#RUN pip install 'transformers[torch]'
#RUN pip install -r requirements.txt


RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install transformers numpy boto3 pillow accelerate
RUN pip install --no-deps diffusers

# Copy function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}
COPY hf_cache ${LAMBDA_TASK_ROOT}/hf_cache

# Set the CMD to your handler
CMD [ "lambda_function.handler" ]