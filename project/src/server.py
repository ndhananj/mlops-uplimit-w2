from fastapi import FastAPI
from ray import serve
from ray.serve.handle import DeploymentHandle
from loguru import logger

from src.data_models import SimpleModelRequest, SimpleModelResponse, SimpleModelResults
from src.model import Model

app = FastAPI(
    title="Drug Review Sentiment Analysis",
    description="Drug Review Sentiment Classifier",
    version="0.1",
)

# TODO: Add in appropriate logging using loguru wherever you see fit
# in order to aid with debugging issues.
logger.add("app.log", rotation="500 MB")

@serve.deployment(
    ray_actor_options={"num_cpus": 0.2},
    autoscaling_config={"min_replicas": 1, "max_replicas": 2},
)
@serve.ingress(app)
class APIIngress:
    def __init__(self, simple_model_handle: DeploymentHandle) -> None:
        self.handle = simple_model_handle

    @app.post("/predict")
    async def predict(self, request: SimpleModelRequest):
        # DONE: Use the handle.predict which is a remote function
        # to get the result
        logger.info(f"Received prediction request for review: {request.review[:50]}...")
        result = await self.handle.predict.remote(request.review)
        logger.info("Prediction completed successfully")
        return SimpleModelResponse.model_validate(result.model_dump())


@serve.deployment(
    ray_actor_options={"num_cpus": 0.2},
    autoscaling_config={"min_replicas": 1, "max_replicas": 2},
)
class SimpleModel:
    def __init__(self) -> None:
        self.session = Model.load_model()

    def predict(self, review: str) -> SimpleModelResults:
        # DONE: Use the Model.predict to get the result
        logger.info(f"Predicting sentiment for review: {review[:50]}...")
        result = self.session.predict(review)
        logger.info("Prediction completed")
        return SimpleModelResults.model_validate(result)


entrypoint = APIIngress.bind(
    SimpleModel.bind(),
)
