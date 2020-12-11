import TensorFlow

struct TitanicModel: Layer {
    var hidden: Dense<Float>
    var dropout: Dropout<Float>
    var output: Dense<Float>
    
    init() {
        hidden = Dense<Float>(
            inputSize: 6,
            outputSize: 25,
            activation: relu)
        
        dropout = Dropout<Float>(probability: 0.25)
        
        output = Dense<Float>(
            inputSize: 25,
            outputSize: 2,
            activation: softmax)
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        input.sequenced(through: hidden, dropout, output)
    }
}
