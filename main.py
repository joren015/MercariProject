import gc

from app.train import main

if __name__ == "__main__":
    main("neural_network")
    gc.collect()
    main("category_model")
    gc.collect()
    main("light_gbm")
