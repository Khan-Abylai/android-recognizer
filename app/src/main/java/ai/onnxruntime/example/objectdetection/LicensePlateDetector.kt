package ai.onnxruntime.example.objectdetection

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.InputStream
import java.nio.FloatBuffer
import java.util.Collections
import kotlin.math.abs

data class DetectorResult(
    var image: Mat?, var licensePlate: LicensePlate?, var detected: Boolean
)

data class LicensePlate(
    var x: Double,
    var y: Double,
    var w: Double,
    var h: Double,
    var x1: Double,
    var y1: Double,
    var x2: Double,
    var y2: Double,
    var x3: Double,
    var y3: Double,
    var x4: Double,
    var y4: Double,
    var prob: Float
)

internal class LicensePlateDetector {

    private val lpPredictionDetectorThreshold: Float = 0.7F

    fun detect(
        inputStream: InputStream, ortEnv: OrtEnvironment, ortSession: OrtSession
    ): DetectorResult {
        val bitmap = BitmapFactory.decodeStream(inputStream)
        inputStream.reset()
        val originalImage = Mat()
        Utils.bitmapToMat(bitmap, originalImage)
        val bgrImage = Mat()
        Imgproc.cvtColor(originalImage, bgrImage, Imgproc.COLOR_BGRA2RGB)
        val resizedImage = Mat()
        Imgproc.resize(bgrImage, resizedImage, Size(512.0, 512.0))

        val normalizedImage = Mat()
        resizedImage.convertTo(normalizedImage, CvType.CV_32FC3, 2.0 / 255, -1.0)

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
        val height = 512
        val width = 512

        val outputBuffer = transposeFlattenBuffer(inputBuffer, depth, height, width)

        val inputTensor = OnnxTensor.createTensor(ortEnv, outputBuffer, tensorShape)
        inputTensor.use {
            // Step 3: call ort inferenceSession run
            val output = ortSession.run(
                Collections.singletonMap("actual_input", inputTensor), setOf("output")
            )

            output.use {
                val rawOutput = (output?.get(0)?.value) as Array<Array<FloatArray>>
                val licensePlates = getLicensePlates(
                    rawOutput, originalImage.cols() / 512.0, originalImage.rows() / 512.0
                )

                if (licensePlates.size != 0) {
                    val licensePlate = nms(licensePlates).first()
                    val plateImage = getPlateImage(licensePlate, bgrImage)
                    return DetectorResult(plateImage, licensePlate, true)
                } else {
                    return DetectorResult(null, null, false)
                }
            }
        }
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

    private fun nms(licensePlateArray: ArrayList<LicensePlate>): List<LicensePlate> {

        return licensePlateArray.sortedWith(compareBy { it.prob })
    }

    private fun getLicensePlates(
        lpPredictions: Array<Array<FloatArray>>, scaleWidth: Double, scaleHeight: Double
    ): ArrayList<LicensePlate> {
        val predictions = lpPredictions[0]
        val plateGridWidth = 512 / 16
        val plateGridHeight = 512 / 16
        val licensePlateArray = arrayListOf<LicensePlate>()

        for (index in 0 until plateGridHeight * plateGridWidth) {
            val prob: Float = abs(predictions[index][12])
            if (prob > lpPredictionDetectorThreshold) {
                val values = predictions[index]
                val x = values[0] * scaleWidth
                val y = values[1] * scaleHeight
                val x1 = values[4] * scaleWidth + x
                val y1 = values[5] * scaleHeight + y

                val x2 = values[6] * scaleWidth + x
                val y2 = values[7] * scaleHeight + y

                val x3 = values[8] * scaleWidth + x
                val y3 = values[9] * scaleHeight + y

                val x4 = values[10] * scaleWidth + x
                val y4 = values[11] * scaleHeight + y

                val h = ((y2 - y1) + (y4 - y3)) / 2
                val w = ((x3 - x1) + (x4 - x2)) / 2
                val licensePlate = LicensePlate(x, y, w, h, x1, y1, x2, y2, x3, y3, x4, y4, prob)
                licensePlateArray.add(licensePlate)
            }
        }
        return licensePlateArray
    }

    private fun getPlateImage(licensePlate: LicensePlate, frame: Mat): Mat {
        val RECT_LP_COORS = listOf(
            Point(0.0, 0.0), Point(0.0, 31.0), Point(127.0, 0.0), Point(127.0, 31.0)
        )

        val SQUARE_LP_COORS = listOf(
            Point(0.0, 0.0), Point(0.0, 63.0), Point(63.0, 0.0), Point(63.0, 63.0)
        )

        val leftTop = Point(licensePlate.x1, licensePlate.y1)
        val leftBottom = Point(licensePlate.x2, licensePlate.y2)
        val rightTop = Point(licensePlate.x3, licensePlate.y3)
        val rightBottom = Point(licensePlate.x4, licensePlate.y4)

        val square: Boolean = (rightTop.x - leftTop.x) / (leftBottom.y - leftTop.y) < 2.6
        val transformationMatrix: Mat
        val lpSize: Size

        if (square) {
            transformationMatrix = Imgproc.getPerspectiveTransform(
                MatOfPoint2f(leftTop, leftBottom, rightTop, rightBottom), MatOfPoint2f(
                    SQUARE_LP_COORS[0], SQUARE_LP_COORS[1], SQUARE_LP_COORS[2], SQUARE_LP_COORS[3]
                )
            )

            lpSize = Size(64.0, 64.0)
        } else {
            transformationMatrix = Imgproc.getPerspectiveTransform(
                MatOfPoint2f(leftTop, leftBottom, rightTop, rightBottom), MatOfPoint2f(
                    RECT_LP_COORS[0], RECT_LP_COORS[1], RECT_LP_COORS[2], RECT_LP_COORS[3]
                )
            )

            lpSize = Size(128.0, 32.0)
        }
        val plateImage = Mat()
        Imgproc.warpPerspective(frame, plateImage, transformationMatrix, lpSize)
        return plateImage
    }
}