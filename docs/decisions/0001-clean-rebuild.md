# ADR 0001: LaflaAi-Core temiz yeniden kurulur

## Durum

Ilk LaflaGPT Colab kosusu modeli yukleyip token uretebildi. Ancak cikti kalitesi
bozuktu: Turkce karakterler, token sinirlari, dusunce bicimi ve sohbet davranisi
guvenilir degildi.

## Karar

Yeni cekirdek `LaflaAi-Core` adiyla sifirdan kurulacak. Eski checkpoint ve eski
kod dogrudan temel alinmayacak.

## Gerekce

- Eski tokenizer Turkce/UTF-8 kalitesini tasimiyor.
- 1000 step sifirdan egitim model kalitesi icin yeterli degil.
- Colab destegi gercek Drive mount, resume, arşivleme ve kota guard disiplini
  olmadan kirilgan kaliyor.
- Mobil ve dusuk guclu CPU hedefleri bastan export/quantization tasarimi istiyor.
- Prompt kurallari tek script ve gevsek dosya yapisini kabul etmiyor.

## Sonuc

Eski checkpoint silinmeyecek ama yeni agirlik temeli olmayacak. Yeni pipeline,
eski checkpointten daha iyi tokenizer raporu, daha iyi eval ve daha temiz Colab
akisiyla baslayacak.

