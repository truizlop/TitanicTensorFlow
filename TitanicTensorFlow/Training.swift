import PythonKit
import TensorFlow

func train(
    model: inout TitanicModel,
    samples: [TitanicBatch],
    epochCount: Int = 1000,
    batchSize: Int = 32
) {
    let epochs = TrainingEpochs(samples: samples, batchSize: batchSize)
    let optimizer = Adam(for: model)
    var trainLoss: [Float] = []
    var trainAccuracy: [Float] = []
    
    for (epochIndex, epoch) in epochs.prefix(epochCount).enumerated() {
        var epochLoss: Float = 0
        var epochAccuracy: Float = 0
        var batchCount: Float = 0
        
        for batchSamples in epoch {
            let batch = batchSamples.collated
            let (loss, gradient) = valueWithGradient(at: model) { model -> Tensor<Float> in
                let logits = model(batch.features)
                return softmaxCrossEntropy(logits: logits, labels: batch.labels)
            }
            optimizer.update(&model, along: gradient)
            
            let logits = model(batch.features)
            epochAccuracy += accuracy(predictions: logits.argmax(squeezingAxis: 1), truths: batch.labels)
            epochLoss += loss.scalarized()
            batchCount += 1
        }
        
        epochAccuracy /= batchCount
        epochLoss /= batchCount
        
        trainAccuracy.append(epochAccuracy)
        trainLoss.append(epochLoss)
        
        if (epochIndex + 1) % 50 == 0 {
            print("Epoch: \(epochIndex + 1), Loss: \(epochLoss), Accuracy: \(epochAccuracy)")
        }
    }
    
    //plot(loss: trainLoss, train: trainAccuracy)
}

func accuracy(
    predictions: Tensor<Int32>,
    truths: Tensor<Int32>
) -> Float {
    Tensor<Float>(predictions .== truths).mean().scalarized()
}

func plot(loss: [Float], train: [Float]) {
    let plt = Python.import("matplotlib.pyplot")
    
    plt.figure(figsize: [12, 8])
    
    let lossAxes = plt.subplot(2, 1, 1)
    lossAxes.set_ylabel("Loss")
    lossAxes.set_xlabel("Epoch")
    lossAxes.plot(loss)
    
    let trainAxes = plt.subplot(2, 1, 2)
    trainAxes.set_ylabel("Accuracy")
    trainAxes.plot(train)
    
    plt.show()
}
