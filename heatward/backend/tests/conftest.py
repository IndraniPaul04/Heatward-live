import os, tempfile
os.environ["WEATHER_SOURCE"] = "demo"
os.environ["CACHE_DIR"] = tempfile.mkdtemp()
os.environ["MIN_MANUAL_REFRESH_S"] = "0"
os.environ["TIERB_PATH"] = os.path.join(tempfile.mkdtemp(), "none.joblib")   # tests must not depend on a locally trained model
