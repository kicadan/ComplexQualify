import json
import tempfile
import boto3
import statistics
import numpy as np
import pickle
from scipy.stats.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier

s3 = boto3.resource('s3')


def lambda_handler(event, context):
    # TODO implement
    s3.Bucket("fall-classificators-bucket").download_file("RF-48", "/tmp/RF-48")
    frameDuration = 6

    wearable = event["measurements"]["wearable"]
    phone = event["measurements"]["phone"]

    xw = []
    yw = []
    zw = []
    xp = []
    yp = []
    zp = []
    magw = []
    magp = []
    # arrays for jerk
    jerkxw = []
    jerkyw = []
    jerkzw = []
    jerkxp = []
    jerkyp = []
    jerkzp = []

    for row in wearable:
        xw.append(int(row["X"]))
        yw.append(int(row["Y"]))
        zw.append(int(row["Z"]))
        magw.append(np.linalg.norm([int(row["X"]), int(row["Y"]), int(row["Z"])]))
    for row in phone:
        xp.append(float(row["X"]))
        yp.append(float(row["Y"]))
        zp.append(float(row["Z"]))
        magp.append(np.linalg.norm([float(row["X"]), float(row["Y"]), float(row["Z"])]))
    roww = list(zip(xw[1:], yw[1:], zw[1:]))
    # no possibility to count for first sample
    x = xw[0]
    y = yw[0]
    z = zw[0]
    timeStepw = float(frameDuration / len(xw))
    timeStepp = float(frameDuration / len(xp))
    for row in roww:
        print(row)
        jerkxw.append(abs((row[0] - x) / (timeStepw)))
        jerkyw.append(abs((row[1] - y) / (timeStepw)))
        jerkzw.append(abs((row[2] - z) / (timeStepw)))
        x = row[0]
        y = row[1]
        z = row[2]
    rowp = list(zip(xp[1:], yp[1:], zp[1:]))
    x = xp[0]
    y = yp[0]
    z = zp[0]
    for row in rowp:
        jerkxp.append(abs((row[0] - x) / (timeStepp)))
        jerkyp.append(abs((row[1] - y) / (timeStepp)))
        jerkzp.append(abs((row[2] - z) / (timeStepp)))
        x = row[0]
        y = row[1]
        z = row[2]
    features = [max(magw), max(magp), pearsonr(xw, yw)[0], pearsonr(xp, yp)[0], pearsonr(xw, zw)[0],
                pearsonr(xp, zp)[0], pearsonr(yw, zw)[0], pearsonr(yp, zp)[0],
                statistics.mean(xw), statistics.mean(xp), statistics.mean(yw), statistics.mean(yp), statistics.mean(zw),
                statistics.mean(zp),
                statistics.variance(xw), statistics.variance(xp), statistics.variance(yw), statistics.variance(yp),
                statistics.variance(zw), statistics.variance(zp),
                statistics.stdev(xw), statistics.stdev(xp), statistics.stdev(yw), statistics.stdev(yp),
                statistics.stdev(zw), statistics.stdev(zp),
                sum(np.absolute(xw)), sum(np.absolute(xp)), sum(np.absolute(yw)), sum(np.absolute(yp)),
                sum(np.absolute(zw)), sum(np.absolute(zp)),
                (max(xw) - min(xw)), (max(xp) - min(xp)), (max(yw) - min(yw)), (max(yp) - min(yp)), (max(zw) - min(zw)),
                (max(zp) - min(zp)),
                max(jerkxw), max(jerkxp), max(jerkyw), max(jerkyp), max(jerkzw), max(jerkzp),
                np.linalg.norm([statistics.stdev(xw), statistics.stdev(zw)]),
                np.linalg.norm([statistics.stdev(xp), statistics.stdev(zp)]),
                np.linalg.norm([statistics.stdev(xw), statistics.stdev(yw), statistics.stdev(zw)]),
                np.linalg.norm([statistics.stdev(xp), statistics.stdev(yp), statistics.stdev(zp)])]

    clf = pickle.load(open("/tmp/RF-48", 'rb'))

    return {
        'result': clf.predict([features])
    }
