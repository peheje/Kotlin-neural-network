// http://iamtrask.github.io/2015/07/12/basic-python-network/

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.factory.Nd4j.create
import org.nd4j.linalg.factory.Nd4j.rand
import org.nd4j.linalg.indexing.INDArrayIndex
import org.nd4j.linalg.ops.transforms.Transforms.*

infix fun INDArray.dot(o: INDArray): INDArray = this.mmul(o)
fun floats(vararg floats: Float) = floatArrayOf(*floats)
fun ints(vararg ints: Int) = intArrayOf(*ints)

fun simpleNet() {
    // var w0 = create(floats(0.1f, -0.1f, -0.2f), ints(3, 1))
    val lr = 0.001
    var w0 = rand(3, 1)
    var b0 = Nd4j.getRandom().nextDouble()
    val x = create(floats(
            0f, 0f, 1f,
            0f, 1f, 1f,
            1f, 0f, 1f,
            1f, 1f, 1f), ints(4, 3))
    val y = create(floats(
            0f,
            1f,
            0f,
            1f
    ), ints(4, 1))
    var l1: INDArray? = null

    for (i in 0 until 10000) {
        val l0 = x
        l1 = sigmoid((l0 dot w0).add(b0))
        val l1err = y.sub(l1)
        val l1delta = sigmoidDerivative(l1).mul(l1err)
        w0.addi(l0.transpose() dot l1delta).mul(lr)
        b0 -= l1delta.sumNumber().toDouble() * lr
    }

    println(l1)
    println(b0)
}
