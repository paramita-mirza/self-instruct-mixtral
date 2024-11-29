import sentencepiece.sentencepiece_model_pb2 as model

mp = model.ModelProto()
model_file = "/raid/s3/opengptx/models/hub/models--EuropeanLLM-Beta--HalloEurope-7B/snapshots/2e9290ab856d1847038241232c6815fe6c6d0de6/tokenizer.model"
symbols = ['<|im_start|>', '<|im_end|>']

mp.ParseFromString(open(model_file, 'rb').read())
print(f'Original model pieces: {len(mp.pieces)}')

for i, sym in enumerate(symbols, 1):
    new_sym = mp.SentencePiece()
    new_sym.piece = sym
    new_sym.score = 0.0  # default score for USER_DEFINED
    new_sym.type = 4  # type value for USER_DEFINED
    mp.pieces.append(new_sym)
    print(f'added {new_sym}...')

print(f'New model pieces: {len(mp.pieces)}')

outfile = '/raid/s3/opengptx/paramita/teuken_tokenizer/tokenizer.model'
with open(outfile, 'wb') as f:
    f.write(mp.SerializeToString())