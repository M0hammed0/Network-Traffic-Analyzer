# Network Traffic Analyzer

## Proje Açıklaması

Network Traffic Analyzer, ağ trafiğini analiz etmek ve anomali tespiti yapmak için geliştirilmiş bir uygulamadır. Bu uygulama, ağ paketlerini yakalar, analiz eder ve kullanıcıya görsel grafikler ve raporlar sunar. Gelişmiş Anomali Tespit Sistemi (AIS) ile entegre edilmiştir ve çeşitli makine öğrenimi algoritmalarını kullanarak anomali tespiti yapar.

## Özellikler

- **Ağ Paket Yakalama**: Gerçek zamanlı olarak ağ paketlerini yakalar.
- **Anomali Tespiti**: K-Means, Isolation Forest ve One-Class SVM gibi algoritmalar kullanarak anomali tespiti yapar.
- **Görsel Grafikler**: Protokol dağılımı, paket boyutu dağılımı ve trafik zaman grafikleri gibi çeşitli grafikler sunar.
- **Rapor Oluşturma**: Kullanıcıların analiz sonuçlarını PDF formatında rapor olarak kaydetmesine olanak tanır.
- **Kullanıcı Arayüzü**: Kullanıcı dostu bir arayüz ile kolay kullanım sağlar.

## Gereksinimler

- Python 3.x
- Gerekli kütüphaneler, `requirements.txt` dosyasında belirtilmiştir.

## Kurulum

1. **Python Yükleyin**: Eğer Python yüklü değilse, [Python'un resmi web sitesinden](https://www.python.org/downloads/) Python'u indirin ve yükleyin.
2. **Proje Dosyalarını İndirin**: Bu projeyi klonlayın veya ZIP dosyası olarak indirin.
   ```bash
   git clone https://github.com/kullanici_adi/network-traffic-analyzer.git
   cd network-traffic-analyzer
   ```
3. **Gerekli Kütüphaneleri Yükleyin**: Aşağıdaki komutu çalıştırarak gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```
4. **Uygulamayı Başlatın**: Aşağıdaki komutu çalıştırarak uygulamayı başlatın:
   ```bash
   python npb.py
   ```

## Kullanım

- Uygulama açıldığında, "Start Analysis" butonuna tıklayarak ağ trafiğini analiz etmeye başlayabilirsiniz.
- Anomaliler tespit edildiğinde, uygulama bu bilgileri kullanıcı arayüzünde gösterecektir.
- "Generate Report" butonuna tıklayarak analiz sonuçlarınızı PDF formatında kaydedebilirsiniz.

## Katkıda Bulunanlar

- [MOHAMMAD AL ZUBİ] - Proje sahibi ve geliştirici

## Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Daha fazla bilgi için LICENSE dosyasına bakın.

## İletişim

Herhangi bir sorunuz veya öneriniz varsa, lütfen [MUHAMMED-PORTFOLİO](https://muhammed-portfoli0.vercel.app/) adresi üzerinden iletişime geçin.
