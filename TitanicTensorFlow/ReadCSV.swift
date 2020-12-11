import Foundation
import PythonKit
import TensorFlow

func readTitanic(file: String) -> [TitanicBatch] {
    let np = Python.import("numpy")
    let filename = URL(fileURLWithPath: #file)
        .deletingLastPathComponent()
        .appendingPathComponent(file)
        .appendingPathExtension("csv")
        .absoluteString.replacingOccurrences(of: "file://", with: "")
    
    let features = np.loadtxt(
        filename,
        delimiter: ";",
        skiprows: 1,
        usecols: [1, 2, 3, 6, 7, 8],
        dtype: Float.numpyScalarTypes.first!)
    
    let labels = np.loadtxt(
        filename,
        delimiter: ";",
        skiprows: 1,
        usecols: [0],
        dtype: Int32.numpyScalarTypes.first!)
    
    guard let featuresTensor = Tensor<Float>(numpy: features),
          let labelsTensor = Tensor<Int32>(numpy: labels) else {
        fatalError("Could not load dataset \(file)")
    }
    
    return zip(
        featuresTensor.unstacked(),
        labelsTensor.unstacked()
    ).map { pair in
        TitanicBatch(features: pair.0, labels: pair.1)
    }
}
