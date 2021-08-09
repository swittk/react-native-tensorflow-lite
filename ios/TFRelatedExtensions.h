//
//  TFRelatedExtensions.h
//  react-native-tensorflow-lite
//
//  Created by Switt Kongdachalert on 31/7/2564 BE.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import "TensorflowLite-Swift.h"
NS_ASSUME_NONNULL_BEGIN

@interface UIImage (TFRelatedExtensions)

/// Creates and returns a new image scaled to the given size. The image preserves its original PNG
/// or JPEG bitmap info.
///
/// - Parameter size: The size to scale the image to.
/// - Returns: The scaled image or `nil` if image could not be resized.
-(UIImage *)scaledImageWithSize:(CGSize)size;
//
///// Returns the data representation of the image after scaling to the given `size` and removing
///// the alpha component. This function assumes a batch size of one and three channels per image.
///// Changing these parameters in the TF Lite model will cause conflicts.
/////
///// - Parameters
/////   - size: Size to scale the image to (i.e. image size used while training the model).
/////   - isQuantized: Whether the model is quantized (i.e. fixed point values rather than floating
/////       point values).
///// - Returns: The scaled image as data or `nil` if the image could not be scaled.
-(NSData *)scaledDataWithSize:(CGSize)size isQuantized:(BOOL)isQuantized;
@end

NS_ASSUME_NONNULL_END
