//
//  TFRelatedExtensions.h
//  react-native-tensorflow-lite
//
//  Created by Switt Kongdachalert on 31/7/2564 BE.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
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
-(NSData *)scaledDataWithSize:(CGSize)size isQuantized:(BOOL)isQuantized grayscale:(BOOL)grayscale;
/**
    The image, resized to fit the size specified, with black bars padding
 */
-(UIImage *)imageFittedToSize:(CGSize)sizeTarget;
-(UIImage *)imageFittedToSize:(CGSize) sizeTarget opaque:(BOOL)opaque scale:(float)scale;
-(UIImage *)imageFittedToSize:(CGSize) sizeTarget opaque:(BOOL)opaque scale:(float)scale backgroundColor:(nullable UIColor *)bgColor;

+(NSData *)normalizedDataFromImage:(CGImageRef)image resizingToSize:(CGSize)size;
+(NSData *)normalizedDataFromImage:(CGImageRef)image resizingToSize:(CGSize)size grayscale:(BOOL)grayscale;
+(CGImageRef)createNormalizeImage:(CGImageRef)image resizingToSize:(CGSize)size;

-(UIImage *)cropToX:(CGFloat)x y:(CGFloat)y width:(CGFloat)width height:(CGFloat)height;
-(UIImage *)relativeCropToX:(CGFloat)x y:(CGFloat)y width:(CGFloat)width height:(CGFloat)height;
@end

NS_ASSUME_NONNULL_END
