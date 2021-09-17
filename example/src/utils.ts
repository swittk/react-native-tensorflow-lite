export function base64ToArrayBuffer(base64: string) {
  const binary_string = global.atob(base64);
  const len = binary_string.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) bytes[i] = binary_string.charCodeAt(i);
  return bytes.buffer;
}

const base64FirstCharMap: Record<string, string> = {
  '/': 'data:image/jpg;base64',
  i: 'data:image/png;base64',
  R: 'data:image/gif;base64',
  P: 'data:image/svg+xml;base64',
  U: 'data:image/webp;base64'
};
const base64TypeMap: Record<string, string> = {
  '/': 'jpg',
  i: 'png',
  R: 'gif',
  P: 'svg+xml',
  U: 'webp'
};
export function base64URIPrefixForBase64Content(content: string): string | undefined {
  const firstChar = content.charAt(0);
  return base64FirstCharMap[firstChar];
}
export function fileTypeForBase64(content: string): string | undefined {
  const firstChar = content.charAt(0);
  return base64TypeMap[firstChar];
}
export function base64RawToDataURI(content: string) {
  const prefix = base64URIPrefixForBase64Content(content);
  return `${prefix || ''},${content}`;
}
