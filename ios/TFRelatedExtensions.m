//
//  TFRelatedExtensions.m
//  react-native-tensorflow-lite
//
//  Created by Switt Kongdachalert on 31/7/2564 BE.
//

#import "TFRelatedExtensions.h"

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
    SafeLog(@"I Should be logged, always");
    if(!self.CGImage) {
        SafeLog(@"I don't have CGIMAGE");
        return nil;
    }
    return [UIImage normalizedDataFromImage:self.CGImage resizingToSize:size];
}
+(CGImageRef)normalizeImage:(CGImageRef)image resizingToSize:(CGSize)size {
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
        CFRelease(colorSpace);
        if(colorspaceSameAsDevice || colorspaceNameIsSRGB) {
            // Return the image if it is in the right format
            // to save a redraw operation.
            CFAutorelease(image);
            return image;
        }
    }
    else {
        CFRelease(colorSpace);
    }
    
    CGBitmapInfo bitmapInfo = kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big;
    int width = size.width;
    unsigned long scaledBytesPerRow = (CGImageGetBytesPerRow(image) / CGImageGetWidth(image)) * width;
    // Create a bitmap context
    CGContextRef context = CGBitmapContextCreate(NULL, width, (int)size.height, CGImageGetBitsPerComponent(image), scaledBytesPerRow, colorSpace, bitmapInfo);
    if(!context) {
        SafeLog(@"Failed to create CGContextRef");
        CGImageRelease(image);
        return nil;
    }
    CGContextDrawImage(context, CGRectMake(0, 0, size.width, size.height), image);
    CGImageRef createdImage = CGBitmapContextCreateImage(context);
    CGContextRelease(context);
    CGImageRelease(image);
    CFAutorelease(createdImage);
    return createdImage;
}

+(NSData *)normalizedDataFromImage:(CGImageRef)image resizingToSize:(CGSize)size {
    CGImageRef normalizedImage = [self normalizeImage:image resizingToSize:size];
    if(!normalizedImage) {
        SafeLog(@"NormalizedImage result was nil");
        return nil;
    }
    CGImageRetain(normalizedImage);
    CGDataProviderRef dataProvider = CGImageGetDataProvider(normalizedImage);
    CFDataRef data = CGDataProviderCopyData(dataProvider);
//    CGDataProviderRelease(dataProvider); // Likely cause of error : "Get"DataProvider = we shouldn't free it.
    CGImageRelease(normalizedImage);
    if(!data) { SafeLog(@"Failed to get data from image"); return nil; };
    // After here we shall not return before the end of the function
    // Because data must be freed.
    
    size_t bitsPerPixel = CGImageGetBitsPerPixel(normalizedImage);
    size_t bitsPerComponent = CGImageGetBitsPerComponent(normalizedImage);
    CGImageAlphaInfo imageAlphaInfo = CGImageGetAlphaInfo(normalizedImage);
    CGBitmapInfo bitmapInfo = CGImageGetBitmapInfo(normalizedImage);
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
            SafeLog([NSString stringWithFormat:@"Outdata16 is %@", outData]);
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
            // We have 3 Float32 channels (RGB) out, while the input was Int32 for RGBA.
            size_t allocLen = sizeof(Float32) * 3 * dataLength / 4;
            NSLog(@"datalen %d, allocating %d", dataLength, allocLen);
            Float32 *outBytes = malloc(allocLen);
            
            Float32 maximumChannelValue = 255; // 2 ^ 8 - 1
            // TF Note:
            // Iterate over channels individually. Since the order of the channels in memory
            // may vary, we cannot add channels to the float buffer we pass to TF Lite in the
            // order that they are iterated over.
            // My Note : Basically just rearrange in each Float32 byte why say it so verbosely..
            NSUInteger outBytesIndex = 0;
            for(NSUInteger byteIndex = 0; byteIndex < dataLength; byteIndex += 4) {
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
            SafeLog([NSString stringWithFormat:@"Outdata32 is %@", outData]);
        } break;
        default:{
            SafeLog(@"Unsupported format from image");
        }
    }
    CFRelease(data);
    return outData;
}


@end

