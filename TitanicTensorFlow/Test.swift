import TensorFlow

func test(model: TitanicModel, with samples: [TitanicBatch]) {
    let batch = samples.collated
    let logits = model(batch.features)
    let predictions = logits.argmax(squeezingAxis: 1)
    let acc = accuracy(predictions: predictions, truths: batch.labels)
    print("Dev batch accuracy: \(acc)")
}
