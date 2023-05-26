package ai.onnxruntime.example.objectdetection

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import java.nio.FloatBuffer
import java.util.Collections
import kotlin.math.exp
import kotlin.math.max

data class RecognizerOutputExtraction(
    var plateLabel: String, var plateProb: Float
)

internal class LicensePlateRecognizer {
    fun recognize(
        plateImg: Mat,
        ortEnv: OrtEnvironment,
        ortSession: OrtSession
    ): RecognizerOutputExtraction {

        val plateSize = plateImg.size()

        var correctImage = Mat()
        if (plateSize == Size(64.0, 64.0)) {
            correctImage = horizontalDivide(plateImg)
        } else {
            plateImg.copyTo(/* m = */ correctImage)
        }

        val normalizedImage = Mat()
        correctImage.convertTo(normalizedImage, CvType.CV_32FC3, 1 / 255.0)

        val inputData = FloatArray(normalizedImage.total().toInt() * 3)
        normalizedImage.get(0, 0, inputData)
        val inputBuffer = FloatBuffer.wrap(inputData)
        val tensorShape = longArrayOf(
            1,
            normalizedImage.channels().toLong(),
            normalizedImage.rows().toLong(),
            normalizedImage.cols().toLong()
        )
        inputBuffer.rewind()
        val depth = 3
        val height = 32
        val width = 128

        val outputBuffer = transposeFlattenBuffer(inputBuffer, depth, height, width)
        val inputTensor = OnnxTensor.createTensor(ortEnv, outputBuffer, tensorShape)

        inputTensor.use {
            val output = ortSession.run(
                Collections.singletonMap("actual_input", inputTensor), setOf("output")
            )

            output.use {
                val rawOutput = (output?.get(0)?.value) as Array<Array<FloatArray>>
                val output = rawOutput[0]
                return extractOutput(output)
            }
        }

    }

    private fun extractOutput(recognizerOutput: Array<FloatArray>): RecognizerOutputExtraction {
        val ALPHABET = "-0123456789abcdefghijklmnopqrstuvwxyz"
        val predictions = recognizerOutput.copyOf()
        for (i in 0 until 30) {
            val tempOutput = FloatArray(37) { 0f }
            for (j in 0 until 37) {
                tempOutput[j] = recognizerOutput[i][j]
            }
            val result = softmax(tempOutput)
            for (k in 0 until 37) {
                predictions[i][k] = result[k]
            }
        }
        var prob = 1.0f
        var currentLabel = String()
        var currentProb = 1.0f
        var currentChar = 0
        for (i in 0 until 30) {
            var maxProb = 0.0f
            var maxIndex = 0

            for (j in 0 until 37) {
                if (maxProb < predictions[i][j]) {
                    maxIndex = j
                    maxProb = predictions[i][j]
                }
            }
            if (maxIndex == currentChar) {
                currentProb = max(maxProb, currentProb)
            } else {
                if (currentChar != 0) {
                    currentLabel += ALPHABET[currentChar]
                    prob *= currentProb
                }
                currentProb = maxProb
                currentChar = maxIndex
            }
        }

        if (currentChar != 0) {
            currentLabel += ALPHABET[currentChar]
            prob *= currentProb
        }

        if (currentLabel.isEmpty()) {
            currentLabel += ALPHABET[0]
            prob = 0.0f
        }

        return RecognizerOutputExtraction(currentLabel, prob)
    }


    private fun softmax(scoreVec: FloatArray): MutableList<Float> {
        val softmaxVec = MutableList(37) { 0.0f }
        val scoreMax = scoreVec.maxOrNull() ?: 0.0f
        var eSum = 0.0
        for (j in 0 until 37) {
            softmaxVec[j] = exp(scoreVec[j] - scoreMax).toFloat()
            eSum += softmaxVec[j]
        }
        for (k in 0 until 37) {
            softmaxVec[k] /= eSum.toFloat()
        }
        return softmaxVec
    }

    private fun horizontalDivide(originalImage: Mat): Mat {
        val roi1 = originalImage.submat(0, 32, 0, 64)
        val roi2 = originalImage.submat(32, 64, 0, 64)

        val outputImage = Mat.zeros(Size(128.0, 32.0), CvType.CV_8UC3)

        roi1.copyTo(outputImage.submat(0, 32, 0, 64))

        roi2.copyTo(outputImage.submat(0, 32, 64, 128))
        return outputImage
    }

    private fun transposeFlattenBuffer(
        buffer: FloatBuffer, depth: Int, height: Int, width: Int
    ): FloatBuffer {
        val outputSize = depth * height * width
        val outputBuffer = FloatBuffer.allocate(outputSize)

        val arr1 = FloatBuffer.allocate(height * width)
        val arr2 = FloatBuffer.allocate(height * width)
        val arr3 = FloatBuffer.allocate(height * width)

        for (z in 0 until depth) {
            for (y in 0 until height) {
                for (x in 0 until width) {
                    val inputIndex = z * height * width + y * width + x
                    if (inputIndex % 3 == 0) {
                        arr1.put(buffer.get(inputIndex))
                    } else if (inputIndex % 3 == 1) {
                        arr2.put(buffer.get(inputIndex))
                    } else {
                        arr3.put(buffer.get(inputIndex))
                    }
                }
            }
        }
        arr1.rewind()
        arr2.rewind()
        arr3.rewind()
        outputBuffer.put(arr1)
        outputBuffer.put(arr2)
        outputBuffer.put(arr3)

        outputBuffer.rewind()

        return outputBuffer
    }
}