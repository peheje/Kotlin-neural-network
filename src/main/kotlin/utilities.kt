import java.util.concurrent.ThreadLocalRandom

fun Double.format(digits: Int) = java.lang.String.format("%.${digits}f", this)
fun stringToByteArray(str: String): ByteArray {
    return kotlin.ByteArray(str.length) { i -> str[i].toByte() }
}

fun byteArrayToString(bytes: ByteArray): String {
    return Array(bytes.size) { i -> bytes[i].toChar() }.joinToString("")
}

fun random(): Double {
    return ThreadLocalRandom.current().nextDouble()
}

fun random(from: Double, to: Double): Double {
    return ThreadLocalRandom.current().nextDouble(from, to)
}

fun linspace(min: Double, max: Double, points: Int): DoubleArray {
    if (points == 1) return DoubleArray(1) { min }
    val d = DoubleArray(points)
    val step = (max - min) / (points - 1)
    for (i in 0 until points) {
        d[i] = min + i * step
    }
    return d
}

fun map(x: Double, originFrom: Double, originTo: Double, from: Double, to: Double): Double {
    //Y = (X-A)/(B-A) * (D-C) + C
    return (x - originFrom) / (originTo - originFrom) * (to - from) + from
}