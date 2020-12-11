import Foundation
import TensorFlow

struct TitanicBatch {
    let features: Tensor<Float>
    let labels: Tensor<Int32>
}

extension TitanicBatch: Collatable {
    init<BatchSamples: Collection>(collating samples: BatchSamples)
    where TitanicBatch == BatchSamples.Element {
        self.features = Tensor<Float>(stacking: samples.map(\.features))
        self.labels = Tensor<Int32>(stacking: samples.map(\.labels))
    }
}
