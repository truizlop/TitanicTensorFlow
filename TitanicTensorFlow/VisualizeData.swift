import PythonKit

func visualize(column a: Int, vs b: Int, in data: [TitanicBatch]) {
    let plt = Python.import("matplotlib.pyplot")
    
    let batch = data.collated.features.transposed()
    let first = batch[a].scalars
    let second = batch[b].scalars
    
    plt.scatter(first, second, batch.array.scalars)
    plt.show()
}
