import Foundation
import TensorFlow

let trainSamples = readTitanic(file: "train-clean")
let devSamples = readTitanic(file: "dev-clean")

var model = TitanicModel()

Context.local.learningPhase = .training
train(model: &model,
      samples: trainSamples)

Context.local.learningPhase = .inference
test(model: model, with: devSamples)
