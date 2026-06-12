# ADR 0002: Ilk Colab checkpoint'i saklanir ama temel alinmaz

## Durum

Ilk buyuk Colab kosusu final checkpoint arsivi olarak Google
Drive'a kaydedildi. Model calisti fakat Turkce ve sohbet kalitesi bozuktu.

## Karar

Arsiv saklanacak. Yeni egitime bu agirliktan devam edilmeyecek.

## Neden Saklanir

- Checkpoint, safetensors, train state ve log akisi icin ilk ispat.
- Colab mount, arsivleme ve geri yukleme senaryolarinda test girdisi.
- Yeni tokenizer/veri/egitim hattinin karsilastirma alt siniri.

## Neden Temel Alinmaz

- Tokenizer mojibake ve Turkce karakter bozulmasi uretti.
- Veri miktari ve egitim suresi kalite icin yetersizdi.
- Ham cikti kimlik ve dusunce bicimi acisindan guvenilir degildi.
- Kotu temel uzerinde devam egitimi, yeni temiz pipeline'i kirletebilir.

## Kabul Kriteri

Yeni LaflaAi-Core kosusu bu ilk checkpointten iyi sayilmak icin sunlari gecmelidir:

- Turkce roundtrip tokenizer testi.
- Basit kimlik SFT testi.
- 20 sabit promptluk regression seti.
- Colab final artifact'in gercek Drive'da gorunmesi.
- Dusuk guclu CPU export paketinin en az bir smoke testi.
