//
//  TFRelatedExtensions.m
//  react-native-tensorflow-lite
//
//  Created by Switt Kongdachalert on 31/7/2564 BE.
//

#import "TFRelatedExtensions.h"

CGSize CGSizeFittingSize(
                         CGSize sizeImage,
                         CGSize sizeTarget
                         )
{
    CGSize ret;
    CGFloat fw;
    CGFloat fh;
    CGFloat f;
    fw = (CGFloat) (sizeTarget.width / sizeImage.width);
    fh = (CGFloat) (sizeTarget.height / sizeImage.height);
    f = fw < fh ? fw : fh;
    ret = (CGSize){
        .width =  sizeImage.width * f,
        .height = sizeImage.height * f
    };
    return ret;
}

void SafeLog(NSString *str) {
    dispatch_async(dispatch_get_main_queue(), ^{
        NSLog(@"%@", str);
    });
}

@implementation UIImage (TFRelatedExtensions)
-(UIImage *)scaledImageWithSize:(CGSize)size  {
    UIGraphicsBeginImageContextWithOptions(size, NO, self.scale);
    [self drawInRect:CGRectMake(0,0, self.size.width, self.size.height)];
    UIImage *image = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    return image;
}
-(NSData *)scaledDataWithSize:(CGSize)size isQuantized:(BOOL)isQuantized {
    //    SafeLog(@"I Should be logged, always");
    return [self scaledDataWithSize:size isQuantized:isQuantized grayscale:NO];
}
-(NSData *)scaledDataWithSize:(CGSize)size isQuantized:(BOOL)isQuantized grayscale:(BOOL)grayscale {
    if(!self.CGImage) {
        SafeLog(@"I don't have CGIMAGE");
        return nil;
    }
    return [UIImage normalizedDataFromImage:self.CGImage resizingToSize:size grayscale:grayscale];
}
/**
 Normalizes and resizes the given image.
 If the image is the same size and colorspace as wanted. the same CGImageRef is RETAINED and returned.
 If a new image is created, it is a new CGImageRef
 
 **(you MUST RELEASE the return value)**.
 */
+(CGImageRef)createNormalizeImage:(CGImageRef)image resizingToSize:(CGSize)size {
    CGImageRetain(image);
    // Don't forget to release the image before ALL returns
    
    // The TF Lite model expects images in the RGB color space.
    // Device-specific RGB color spaces should have the same number of colors as the standard
    // RGB color space so we probably don't have to redraw them.
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGSize cgImageSize = CGSizeMake(CGImageGetWidth(image), CGImageGetHeight(image));
    if(CGSizeEqualToSize(cgImageSize, size)) {
        CFStringRef imageColorspace = CGColorSpaceCopyName(CGImageGetColorSpace(image));
        CFStringRef deviceColorSpaceName = CGColorSpaceCopyName(colorSpace);
        
        BOOL colorspaceSameAsDevice = CFStringCompare(imageColorspace, deviceColorSpaceName, 0) == kCFCompareEqualTo;
        BOOL colorspaceNameIsSRGB = CFStringCompare(imageColorspace, kCGColorSpaceSRGB, 0) == kCFCompareEqualTo;
        CFRelease(imageColorspace);
        CFRelease(deviceColorSpaceName);
        if(colorspaceSameAsDevice || colorspaceNameIsSRGB) {
            CFRelease(colorSpace);
            // Return the image if it is in the right format
            // to save a redraw operation.
            return image;
        }
    }
    else {
        // nothing
    }
    
    CGBitmapInfo bitmapInfo = kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big;
    int width = size.width;
    unsigned long scaledBytesPerRow = (CGImageGetBytesPerRow(image) / cgImageSize.width) * width;
    // Create a bitmap context
    // There is currently an error occuring when using images from screenshots, CGBitmapContextCreate errors when using screenshots
    // When CGBITMAP_CONTEXT_LOG_ERRORS environment variable is set to 1, the following shows.
    /*
     CGBitmapContextCreate: unsupported parameter combination:
     16 bits/component; integer;
     64 bits/pixel;
     RGB color space model; kCGImageAlphaPremultipliedLast;
     kCGImageByteOrder32Big byte order;
     1536 bytes/row.
     Valid parameters for RGB color space model are:
     16  bits per pixel,         5  bits per component,         kCGImageAlphaNoneSkipFirst
     32  bits per pixel,         8  bits per component,         kCGImageAlphaNoneSkipFirst
     32  bits per pixel,         8  bits per component,         kCGImageAlphaNoneSkipLast
     32  bits per pixel,         8  bits per component,         kCGImageAlphaPremultipliedFirst
     32  bits per pixel,         8  bits per component,         kCGImageAlphaPremultipliedLast
     32  bits per pixel,         10 bits per component,         kCGImageAlphaNone|kCGImagePixelFormatRGBCIF10
     64  bits per pixel,         16 bits per component,         kCGImageAlphaPremultipliedLast
     64  bits per pixel,         16 bits per component,         kCGImageAlphaNoneSkipLast
     64  bits per pixel,         16 bits per component,         kCGImageAlphaPremultipliedLast|kCGBitmapFloatComponents|kCGImageByteOrder16Little
     64  bits per pixel,         16 bits per component,         kCGImageAlphaNoneSkipLast|kCGBitmapFloatComponents|kCGImageByteOrder16Little
     128 bits per pixel,         32 bits per component,         kCGImageAlphaPremultipliedLast|kCGBitmapFloatComponents
     128 bits per pixel,         32 bits per component,         kCGImageAlphaNoneSkipLast|kCGBitmapFloatComponents
     
     valid byte order flags are kCGBitmapByteOrderDefault, kCGBitmapByteOrder16Big, kCGBitmapByteOrder16Little See Quartz 2D Programming Guide (available online) for more information.
     
     */
    CGContextRef context = CGBitmapContextCreate(NULL, width, (int)size.height, CGImageGetBitsPerComponent(image), scaledBytesPerRow, colorSpace, bitmapInfo);
    CFRelease(colorSpace);
    if(!context) {
        SafeLog(@"Failed to create CGContextRef");
        NSLog(@"Image BytesPerRow %d Size %@", scaledBytesPerRow, NSStringFromCGSize(cgImageSize));
        CGImageRelease(image);
        return nil;
    }
    CGContextDrawImage(context, CGRectMake(0, 0, size.width, size.height), image);
    CGImageRef createdImage = CGBitmapContextCreateImage(context);
    CGContextRelease(context);
    CGImageRelease(image);
    return createdImage;
}

+(NSData *)normalizedDataFromImage:(CGImageRef)image resizingToSize:(CGSize)size {
    return [self normalizedDataFromImage:image resizingToSize:size grayscale:NO];
}
+(NSData *)normalizedDataFromImage:(CGImageRef)image resizingToSize:(CGSize)size grayscale:(BOOL)grayscale {
    CGImageRef normalizedImage = [self createNormalizeImage:image resizingToSize:size];
    if(!normalizedImage) {
        SafeLog(@"NormalizedImage result was nil");
        return nil;
    }
//    NSLog(@"NormalizedImageSize %zu, %zu", CGImageGetWidth(normalizedImage), CGImageGetHeight(normalizedImage)); Size is correct
    CGDataProviderRef dataProvider = CGImageGetDataProvider(normalizedImage);
    CFDataRef data = CGDataProviderCopyData(dataProvider);
    //    CGDataProviderRelease(dataProvider); // Likely cause of error : "Get"DataProvider = we shouldn't free it.
    if(!data) {
        SafeLog(@"Failed to get data from image");
        CGImageRelease(normalizedImage);
        return nil;
    };
    // After here we shall not return before the end of the function
    // Because data must be freed.
    size_t bytesPerRow = CGImageGetBytesPerRow(normalizedImage);
    size_t bitsPerPixel = CGImageGetBitsPerPixel(normalizedImage);
    size_t bitsPerComponent = CGImageGetBitsPerComponent(normalizedImage);
    CGImageAlphaInfo imageAlphaInfo = CGImageGetAlphaInfo(normalizedImage);
    CGBitmapInfo bitmapInfo = CGImageGetBitmapInfo(normalizedImage);
    size_t imageHeight = CGImageGetHeight(normalizedImage);
    size_t imageWidth = CGImageGetWidth(normalizedImage);
    CGImageRelease(normalizedImage);
    
    NSData *outData = nil;
    // TF Lite expects an array of pixels in the form of floats normalized between 0 and 1.
    //    var floatArray: [T]
    
    // A full list of pixel formats is listed in this document under Table 2-1:
    // https://developer.apple.com/library/archive/documentation/GraphicsImaging/Conceptual/drawingwithquartz2d/dq_context/dq_context.html#//apple_ref/doc/uid/TP30001066-CH203-CJBHBFFE
    // This code only handles pixel formats supported on iOS in the RGB color space.
    // If you're targeting macOS or macOS via Catalyst, you should support the macOS
    // pixel formats as well.
    switch (bitsPerPixel) {
            
            // 16-bit pixel with no alpha channel. On iOS, this must have 5 bits per channel and
            // no alpha channel. The most significant bits are skipped.
        case 16: {
            if(bitsPerComponent != 5) { SafeLog(@"BitsPerComponent not 5"); break;}
            if((imageAlphaInfo & kCGImageAlphaNoneSkipFirst) == 0) { SafeLog(@"imageAlphaInfo does not have kCGImageAlphaNoneSkipFirst"); break;}
            
            // If this bool is false, assume little endian byte order.
            /** The endianness of this machine */
            BOOL bigEndian;
            BOOL hasLittleEndian = (bitmapInfo & kCGBitmapByteOrder16Little) != 0;
            BOOL hasBigEndian = (bitmapInfo & kCGBitmapByteOrder16Big) != 0;
            if(!(hasLittleEndian && hasBigEndian)) {
                bigEndian = YES;
            }
            else {
                CFByteOrder currentByteOrder = CFByteOrderGetCurrent();
                switch(currentByteOrder) {
                    case CFByteOrderLittleEndian: {
                        bigEndian = YES;
                    } break;
                    case CFByteOrderBigEndian: {
                        bigEndian = NO;
                    } break;
                    default: {
                        // For unknown endianness, assume little endian since it's how most
                        // Apple platforms are laid out nowadays.
                        bigEndian = NO;
                    } break;
                }
            }
            CFIndex dataLength = CFDataGetLength(data);
            UInt16 redMask   = 0b0111110000000000;
            UInt16 greenMask  = 0b0000001111100000;
            UInt16 blueMask   = 0b0000000000011111;

            // Note : I wrote the color output thing first...
            if(!grayscale) {
            // We allocate 3/2 of the data length because..
            //  1. The original data is 16 bits for RGB color (5 bits per RGB channel)
            // 2. Tensorflow expects Float32 data out in format of RGB (Float32 per channel)
            //  That means == 3 * sizeof(Float32) per 2 bytes.
            size_t totalAlloc = sizeof(Float32) * dataLength * 3/2;
            Float32 *outBytes = malloc(totalAlloc);
            
            for(NSUInteger byteIndex = 0; byteIndex < dataLength; byteIndex = byteIndex + 2) {
                CFRange pixelRange = CFRangeMake(byteIndex, byteIndex + 1);
                UInt8 rawPixel[2];
                UInt16 pixel;
                CFDataGetBytes(data, pixelRange, rawPixel);
                if(bigEndian) {
                    // Big endian = More significant bytes come first
                    pixel = ((UInt16)rawPixel[0] << 8) | rawPixel[1];
                }
                else {
                    // Little endian = Less significant bytes come first
                    pixel = ((UInt16)rawPixel[1] << 8) | rawPixel[0];
                }
                UInt16 redChannel   = ((pixel & redMask) >> 10);
                UInt16 greenChannel = ((pixel & greenMask) >> 5);
                UInt16 blueChannel  = ((pixel & blueMask) >> 0);
                
                Float32 maximumChannelValue = 31; // 2 ^ 5 - 1
                Float32 red   = (Float32)(redChannel) / maximumChannelValue;
                Float32 green = (Float32)(greenChannel) / maximumChannelValue;
                Float32 blue  = (Float32)(blueChannel) / maximumChannelValue;
                
                NSUInteger pixelIndex = byteIndex / 2;
                NSUInteger floatIndex = pixelIndex * 3;
                outBytes[floatIndex] = red;
                outBytes[floatIndex + 1] = green;
                outBytes[floatIndex + 2] = blue;
            }
            // Note From Tensorflow implementation:
            // We discard the image's alpha channel before running the TF Lite model, so we can treat
            // alpha and non-alpha images identically.
            outData = [NSData dataWithBytesNoCopy:outBytes length:totalAlloc];
            //            SafeLog([NSString stringWithFormat:@"Outdata16 is %@", outData]);
            }
            else {
                // Is grayscale output
            // We allocate 1/2 of the data length because..
            //  1. The original data is 16 bits for RGB color (5 bits per RGB channel)
            // 2. We expects Float32 data out in format of grayscale (single Float32 channel)
            //  That means == 1 * sizeof(Float32) per 2 bytes.
            size_t totalAlloc = sizeof(Float32) * dataLength * 1/2;
            Float32 *outBytes = malloc(totalAlloc);
            
            for(NSUInteger byteIndex = 0; byteIndex < dataLength; byteIndex = byteIndex + 2) {
                CFRange pixelRange = CFRangeMake(byteIndex, byteIndex + 1);
                UInt8 rawPixel[2];
                UInt16 pixel;
                CFDataGetBytes(data, pixelRange, rawPixel);
                if(bigEndian) {
                    // Big endian = More significant bytes come first
                    pixel = ((UInt16)rawPixel[0] << 8) | rawPixel[1];
                }
                else {
                    // Little endian = Less significant bytes come first
                    pixel = ((UInt16)rawPixel[1] << 8) | rawPixel[0];
                }
                UInt16 redChannel   = ((pixel & redMask) >> 10);
                UInt16 greenChannel = ((pixel & greenMask) >> 5);
                UInt16 blueChannel  = ((pixel & blueMask) >> 0);
                
                Float32 maximumChannelValue = 31; // 2 ^ 5 - 1
                Float32 red   = (Float32)(redChannel) / maximumChannelValue;
                Float32 green = (Float32)(greenChannel) / maximumChannelValue;
                Float32 blue  = (Float32)(blueChannel) / maximumChannelValue;
                
                NSUInteger pixelIndex = byteIndex / 2;
                NSUInteger floatIndex = pixelIndex;
                // grayscale = R * 0.2126 + G * 0.7152 + B * 0.0722
                outBytes[floatIndex] = red * 0.2126 + green * 0.7152 + blue * 0.0722;
            }
            // Note From Tensorflow implementation:
            // We discard the image's alpha channel before running the TF Lite model, so we can treat
            // alpha and non-alpha images identically.
            outData = [NSData dataWithBytesNoCopy:outBytes length:totalAlloc];
            }
        } break;
        case 32: {
            if(bitsPerComponent != 8){ SafeLog(@"BitsPerComponent not 8"); break;}
            BOOL alphaFirst = (
                               imageAlphaInfo == kCGImageAlphaNoneSkipFirst
                               || imageAlphaInfo == kCGImageAlphaPremultipliedFirst
                               );
            BOOL alphaLast = (
                              imageAlphaInfo == kCGImageAlphaNoneSkipLast
                              || imageAlphaInfo == kCGImageAlphaPremultipliedLast
                              );
            
            /** The endianness of this machine */
            BOOL bigEndian = ((bitmapInfo & kCGBitmapByteOrder32Big) != 0);
            BOOL littleEndian = ((bitmapInfo & kCGBitmapByteOrder32Little) != 0);
            if(!(alphaFirst || alphaLast)){ SafeLog(@"Could not deterine Image alpha format"); break;}
            if(!(bigEndian || littleEndian)) { SafeLog(@"Could not determine endianness of image"); break;}
            //            int numberOfChannels = 4;
            UInt8 alphaOffset, redOffset, greenOffset, blueOffset;
            if(bigEndian) {
                alphaOffset = alphaFirst ? 0 : 3;
                redOffset = alphaFirst ? 1 : 0;
                greenOffset = alphaFirst ? 2 : 1;
                blueOffset = alphaFirst ? 3 : 2;
            }
            else {
                alphaOffset = alphaFirst ? 3 : 0;
                redOffset = alphaFirst ? 2 : 3;
                greenOffset = alphaFirst ? 1 : 2;
                blueOffset = alphaFirst ? 1 : 0;
            }
            
            CFIndex dataLength = CFDataGetLength(data);
            size_t expectedByteWidth = imageWidth * 4;
            // The number of bytes to skip once end of row is reached
            size_t numToSkip = 0;
            if(bytesPerRow != expectedByteWidth) {
                // Using dataLength directly sometimes doesn't work, in the case of on-device manipulation,
                // sometimes there are extra `buffer pixels` appended to the end of each row. (data is all zeros, and isn't ever present in normal use of UIImage/CGImage)
                numToSkip = bytesPerRow - expectedByteWidth;
//                NSLog(@"needs padding by %lu bytes", numToSkip);
            }
            CFIndex actualDataLength = (bytesPerRow - numToSkip) * imageHeight;

            // Note : I wrote the color output thing first...
            if(!grayscale) {
                // RGB output (default)
            // We have 3 Float32 channels (RGB) out, while the input was Int32 for RGBA.
            size_t allocLen = sizeof(Float32) * 3 * actualDataLength / 4; // Use bytes
            Float32 *outBytes = malloc(allocLen);
            
            Float32 maximumChannelValue = 255; // 2 ^ 8 - 1
            // TF Note:
            // Iterate over channels individually. Since the order of the channels in memory
            // may vary, we cannot add channels to the float buffer we pass to TF Lite in the
            // order that they are iterated over.
            // My Note : Basically just rearrange in each Float32 byte why say it so verbosely..
            NSUInteger outBytesIndex = 0;
            for(NSUInteger byteIndex = 0; byteIndex < dataLength; byteIndex += 4) {
                if(numToSkip != 0 && (byteIndex + numToSkip) % bytesPerRow == 0) { // If at end of row... (check numToSkip first because modulo is expensive)
                    byteIndex += numToSkip; // add skip bytes if needed
                }
                Float32 red, green, blue;
                UInt8 channelData[4];
                CFDataGetBytes(data, CFRangeMake(byteIndex, 4), channelData);
                red = (Float32)channelData[redOffset]/maximumChannelValue;
                green = (Float32)channelData[greenOffset]/maximumChannelValue;
                blue = (Float32)channelData[blueOffset]/maximumChannelValue;
                // Ignore alpha; useless for us lmao
                
                outBytes[outBytesIndex] = red;
                outBytes[outBytesIndex + 1] = green;
                outBytes[outBytesIndex + 2] = blue;
                outBytesIndex += 3;
            }
            outData = [NSData dataWithBytesNoCopy:outBytes length:allocLen];
            //            SafeLog([NSString stringWithFormat:@"Outdata32 is %@", outData]);
            }
            else {
                // We have 1 Float32 channel (grayscale) out, while the input was Int32 for RGBA.
            size_t allocLen = sizeof(Float32) * 1 * actualDataLength / 4; // Use bytes
            Float32 *outBytes = malloc(allocLen);
            
            Float32 maximumChannelValue = 255; // 2 ^ 8 - 1
            // TF Note:
            // Iterate over channels individually. Since the order of the channels in memory
            // may vary, we cannot add channels to the float buffer we pass to TF Lite in the
            // order that they are iterated over.
            // My Note : Basically just rearrange in each Float32 byte why say it so verbosely..
            NSUInteger outBytesIndex = 0;
            for(NSUInteger byteIndex = 0; byteIndex < dataLength; byteIndex += 4) {
                if(numToSkip != 0 && (byteIndex + numToSkip) % bytesPerRow == 0) { // If at end of row... (check numToSkip first because modulo is expensive)
                    byteIndex += numToSkip; // add skip bytes if needed
                }
                Float32 red, green, blue;
                UInt8 channelData[4];
                CFDataGetBytes(data, CFRangeMake(byteIndex, 4), channelData);
                red = (Float32)channelData[redOffset]/maximumChannelValue;
                green = (Float32)channelData[greenOffset]/maximumChannelValue;
                blue = (Float32)channelData[blueOffset]/maximumChannelValue;
                // Ignore alpha; useless for us lmao

                // grayscale = R * 0.2126 + G * 0.7152 + B * 0.0722
                outBytes[outBytesIndex] = red * 0.2126 + green * 0.7152 + blue * 0.0722;
                outBytesIndex += 1;
            }
            outData = [NSData dataWithBytesNoCopy:outBytes length:allocLen];
            }
        } break;
        default:{
            SafeLog(@"Unsupported format from image");
        } break;
    }
    CFRelease(data);
    return outData;
}

-(UIImage *)imageFittedToSize:(CGSize)sizeTarget {
    return [self imageFittedToSize:sizeTarget opaque:YES scale:1.0 backgroundColor:[UIColor blackColor]];
}
-(UIImage *)imageFittedToSize:(CGSize)sizeTarget opaque:(BOOL)opaque scale:(float)scale {
    return [self imageFittedToSize:sizeTarget opaque:opaque scale:scale backgroundColor:nil];
}
-(UIImage *)imageFittedToSize:(CGSize)sizeTarget opaque:(BOOL)opaque scale:(float)scale backgroundColor:(nullable UIColor *)bgColor
{
    CGSize sizeNewImage;
    CGSize size = self.size;
    UIImage *ret;
    sizeNewImage = CGSizeFittingSize(size, sizeTarget);
    UIGraphicsBeginImageContextWithOptions(sizeTarget, opaque, scale);
    CGContextRef context = UIGraphicsGetCurrentContext();
    if(bgColor) {
        CGContextSetFillColorWithColor(context, bgColor.CGColor);
        CGContextFillRect(context, (CGRect){.origin = CGPointZero, .size = sizeTarget });
    }
    // These two lines convert coordinate system to top left
//    CGContextScaleCTM(context, 1, -1);
//    CGContextTranslateCTM(context, 0, -sizeNewImage.height);
    CGFloat originX = (sizeTarget.width - sizeNewImage.width)/2.0;
    CGFloat originY = (sizeTarget.height - sizeNewImage.height)/2.0;
    CGRect drawToRect = CGRectMake(originX, originY, sizeNewImage.width, sizeNewImage.height);
//    NSLog(@"drawToRect %@", NSStringFromCGRect(drawToRect));
    [self drawImage:[self orientationUpImage] inRect:drawToRect context:context];
//    CGContextDrawImage(context, drawToRect, [self orientationUpImage].CGImage);
    ret = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    return ret;
}
-(UIImage *)cropToRect:(CGRect)rect {
    rect = CGRectMake(rect.origin.x*self.scale,
                      rect.origin.y*self.scale,
                      rect.size.width*self.scale,
                      rect.size.height*self.scale);
    
    CGImageRef imageRef = CGImageCreateWithImageInRect([self orientationUpImage].CGImage, rect);
    UIImage *result = [UIImage imageWithCGImage:imageRef
                                          scale:self.scale
                                    orientation:UIImageOrientationUp];
    CGImageRelease(imageRef);
    return result;
}
-(UIImage *)cropToX:(CGFloat)x y:(CGFloat)y width:(CGFloat)width height:(CGFloat)height {
    CGRect rect = CGRectMake(x*self.scale,
                             y*self.scale,
                             width*self.scale,
                             height*self.scale);
    CGImageRef imageRef = CGImageCreateWithImageInRect([self orientationUpImage].CGImage, rect);
    UIImage *result = [UIImage imageWithCGImage:imageRef
                                          scale:self.scale
                                    orientation:UIImageOrientationUp];
    CGImageRelease(imageRef);
    return result;
}
-(UIImage *)relativeCropToX:(CGFloat)x y:(CGFloat)y width:(CGFloat)width height:(CGFloat)height {
    // UIImage's `size` property is already for its current Orientation.
    CGSize size = self.size;
    return [self cropToX:x*size.width y:y*size.height width:width*size.width height:height*size.height];
}

// Adapted from https://stackoverflow.com/questions/506622/cgcontextdrawimage-draws-image-upside-down-when-passed-uiimage-cgimage/35483967#35483967
-(void)drawImage:(UIImage *)image inRect:(CGRect)rect context:(CGContextRef)context {
    //flip coords
    CGFloat ty = (rect.origin.y + rect.size.height);
    CGContextTranslateCTM(context, 0, ty);
    CGContextScaleCTM(context, 1.0, -1.0);
    
    //draw image
    CGRect rect__y_zero = CGRectMake(rect.origin.x, 0, rect.size.width, rect.size.height);
    CGContextDrawImage(context, rect__y_zero, image.CGImage);
    
    //flip back
    CGContextScaleCTM(context, 1.0, -1.0);
    CGContextTranslateCTM(context, 0, -ty);
}

//ref: http://stackoverflow.com/a/25293588/2298002
+ (UIImage *)cropImage:(UIImage*)image inRect:(CGRect)rect
{
    double (^rad)(double) = ^(double deg) {
        return deg / 180.0 * M_PI;
    };
    
    CGAffineTransform rectTransform;
    switch (image.imageOrientation) {
        case UIImageOrientationLeft:
            rectTransform = CGAffineTransformTranslate(CGAffineTransformMakeRotation(rad(90)), 0, -image.size.height);
            break;
        case UIImageOrientationRight:
            rectTransform = CGAffineTransformTranslate(CGAffineTransformMakeRotation(rad(-90)), -image.size.width, 0);
            break;
        case UIImageOrientationDown:
            rectTransform = CGAffineTransformTranslate(CGAffineTransformMakeRotation(rad(-180)), -image.size.width, -image.size.height);
            break;
        default:
            rectTransform = CGAffineTransformIdentity;
    };
    rectTransform = CGAffineTransformScale(rectTransform, image.scale, image.scale);
    
    CGImageRef imageRef = CGImageCreateWithImageInRect([image CGImage], CGRectApplyAffineTransform(rect, rectTransform));
    UIImage *result = [UIImage imageWithCGImage:imageRef scale:image.scale orientation:image.imageOrientation];
    CGImageRelease(imageRef);
    
    return result;
}
-(UIImage *)orientationUpImage {
    if(self.imageOrientation == UIImageOrientationUp) {
        return self;
    }
    CGSize size = self.size;
    if(self.scale != 1) {
        size.width *= self.scale;
        size.height *= self.scale;
    }
    UIGraphicsBeginImageContext(size);
    [self drawInRect:CGRectMake(0,0,size.width,size.height)];
    UIImage* newImage = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    return newImage;
}
+ (UIImage*)imageFromImage:(UIImage*)image scaledToSize:(CGSize)newSize
{
    UIGraphicsBeginImageContext( newSize );
    [image drawInRect:CGRectMake(0,0,newSize.width,newSize.height)];
    UIImage* newImage = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    
    return newImage;
}

// Adapted from https://stackoverflow.com/posts/41629535
-(CGImageRef)CGImageWithCorrectOrientation {
    CGImageRef myImageRef = self.CGImage;
    if (self.imageOrientation == UIImageOrientationUp) {
        return myImageRef;
    }
    
    CGAffineTransform transform = CGAffineTransformIdentity;
    switch (self.imageOrientation) {
        case UIImageOrientationRight:
        case UIImageOrientationRightMirrored:
            transform = CGAffineTransformTranslate(transform, 0, self.size.height);
            transform = CGAffineTransformRotate(transform, -1.0 * M_PI_2);
            break;
        case UIImageOrientationLeft:
        case UIImageOrientationLeftMirrored:
            transform = CGAffineTransformTranslate(transform, self.size.width, 0);
            transform = CGAffineTransformRotate(transform, M_PI_2);
            break;
        case UIImageOrientationDown:
        case UIImageOrientationDownMirrored:
            transform = CGAffineTransformTranslate(transform, self.size.width, self.size.height);
            transform = CGAffineTransformRotate(transform, M_PI_2);
            break;
        default:
            break;
    }
    
    switch (self.imageOrientation) {
        case UIImageOrientationRightMirrored:
        case UIImageOrientationLeftMirrored:
            transform = CGAffineTransformTranslate(transform, self.size.height, 0);
            transform = CGAffineTransformScale(transform, -1, 1);
            break;
        case UIImageOrientationDownMirrored:
        case UIImageOrientationUpMirrored:
            transform = CGAffineTransformTranslate(transform, self.size.width, 0);
            transform = CGAffineTransformScale(transform, -1, 1);
            break;
        default:
            break;
    }
    
    size_t contextWidth;
    size_t contextHeight;
    switch (self.imageOrientation) {
        case UIImageOrientationRightMirrored:
        case UIImageOrientationLeftMirrored:
        case UIImageOrientationRight:
        case UIImageOrientationLeft:
            contextWidth = CGImageGetHeight(myImageRef);
            contextHeight = CGImageGetWidth(myImageRef);
            break;
        default:
            contextWidth = CGImageGetWidth(myImageRef);
            contextHeight = CGImageGetHeight(myImageRef);
            break;
    }
    
    CGContextRef context = CGBitmapContextCreateWithData(NULL, contextWidth, contextHeight, CGImageGetBitsPerComponent(myImageRef), CGImageGetBytesPerRow(myImageRef), CGImageGetColorSpace(myImageRef), CGImageGetBitmapInfo(myImageRef), NULL, NULL);
    CGContextConcatCTM(context, transform);
    CGContextDrawImage(context, CGRectMake(0, 0, contextWidth, contextHeight), myImageRef);
    CGImageRef outImage = CGBitmapContextCreateImage(context);
    CFAutorelease(outImage);
    return outImage;
}
/**
 Create CGBitmapContext with the same size, width, and properties for the given CGImageRef
 */
-(CGContextRef)CreateCGBitmapContextFromCGImageRef:(CGImageRef)imageRef {
    CGContextRef context = CGBitmapContextCreateWithData(NULL, CGImageGetWidth(imageRef), CGImageGetHeight(imageRef), CGImageGetBitsPerComponent(imageRef), CGImageGetBytesPerRow(imageRef), CGImageGetColorSpace(imageRef), CGImageGetBitmapInfo(imageRef), NULL, NULL);
    return context;
}

@end

