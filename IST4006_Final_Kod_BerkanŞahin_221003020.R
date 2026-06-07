# =============================================================================
# İST 4006 FİNAL PROJESİ - UÇTAN UCA VERİ ANALİZİ VE MODELLEME
# Veri Seti: 2018 ABD Uçuş Gecikmeleri (BTS)
# Bağımlı Değişken: ARR_DELAY (Varış Gecikmesi - Regresyon)
# =============================================================================
#  Veri setinde bulunan kategorik değişkenler (havayolu kodu, havalimanı bilgileri vb.) 
# çok yüksek kardinaliteye sahip olduğundan ve çalışmanın odak noktası gecikme süreleri 
# olduğundan modelleme sürecine dahil edilmemiştir. One-hot encoding yöntemi bu yüzden
# kodlar arasında yoktur.
# ─────────────────────────────────────────────────────────────────────────────
# PAKET KURULUMU VE YÜKLEMESİ
# ─────────────────────────────────────────────────────────────────────────────

paketler <- c("corrplot", "ggplot2", "car", "rpart", "rpart.plot", "gridExtra",
              "caret", "e1071", "randomForest", "gbm", "glmnet",
              "MASS", "nnet", "doParallel", "dplyr", "reshape2","readr")

for (p in paketler) {
  if (!require(p, character.only = TRUE)) {
    install.packages(p, type = "binary")
    library(p, character.only = TRUE)
  }
}

# Paralel işleme (modelleme hızı için — isteğe bağlı, kapatmak için yorum satırı yapın)
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM I: VERİ KEŞFİ VE ÖN İŞLEME (PREPROCESSING)
# ─────────────────────────────────────────────────────────────────────────────

# --- 1.1 Verinin Okunması ve Temel Yapının İncelenmesi ----------------------

dropbox_link <- "https://www.dropbox.com/scl/fi/wk5uojf24s0wj2fvzyhls/2018.csv?rlkey=x5vegijfmcsrgj7vegi0pymvy&dl=1"
flights <- read_csv(dropbox_link)

# Kalkış gecikmesi 1 dakika ve üzeri olan gözlemler alınıyor
flightsnozero <- subset(flights, DEP_DELAY >= 1)

# Kullanılacak sütunların tanımlanması
sebep_sutunlari <- c("CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY",
                     "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY")

sayisal_veri <- c("ARR_DELAY", "DEP_DELAY", "DISTANCE", "AIR_TIME",
                  "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY",
                  "LATE_AIRCRAFT_DELAY")

# NA olan gecikme sebebi sütunları 0 ile doldur
# (Gecikme yoksa o sebebin katkısı 0 dakikadır — alan mantığı)
flightsnozero[, sebep_sutunlari][is.na(flightsnozero[, sebep_sutunlari])] <- 0
flightsnozero[, sayisal_veri][is.na(flightsnozero[, sayisal_veri])]       <- 0

# --- 1.2 Tanımlayıcı İstatistikler ------------------------------------------

cat("\n========== TANIMLAYICI İSTATİSTİKLER ==========\n")
print(summary(flightsnozero[, sayisal_veri]))

cat("\n--- Standart Sapmalar ---\n")
print(sapply(flightsnozero[, sayisal_veri], sd, na.rm = TRUE))

cat("\n--- Gözlem ve Değişken Sayısı ---\n")
cat("Satir:", nrow(flightsnozero), "| Sutun:", ncol(flightsnozero), "\n")

# --- 1.3 Eksik Veri Analizi -------------------------------------------------

cat("\n========== EKSİK VERİ ANALİZİ ==========\n")
eksik_sayisi <- colSums(is.na(flightsnozero[, sayisal_veri]))
eksik_oran   <- round(eksik_sayisi / nrow(flightsnozero) * 100, 2)
eksik_tablo  <- data.frame(Eksik_Sayi = eksik_sayisi, Eksik_Oran_Yuzde = eksik_oran)
print(eksik_tablo)
# Not: Yukarıda NA → 0 dönüşümü yapıldığından bu aşamada kayıp kalmaması beklenir.

# --- 1.4 Örnekleme (Büyük Veri Seti için) ------------------------------------

set.seed(123)
flights_sample <- flightsnozero[sample(nrow(flightsnozero), 10000), ]

# --- 1.5 Aykırı Değer Analizi (Box-Plot) ------------------------------------

# Sayısal değişkenler için boxplot — aykırı gözlemleri görsel olarak tespit eder
par(mfrow = c(2, 4), mar = c(4, 4, 2, 1))
for (v in sayisal_veri) {
  boxplot(flights_sample[[v]],
          main  = v,
          col   = "steelblue",
          border = "navy",
          outline = TRUE,
          ylab  = "Dakika")
}
par(mfrow = c(1, 1))

# Aykırı değer müdahalesi: IQR yöntemiyle uç gözlemler kırpılıyor (winsorizing)
# ARR_DELAY için: alt sınır = Q1 - 1.5*IQR, üst sınır = Q3 + 1.5*IQR
iqr_kirpma <- function(x) {
  Q1 <- quantile(x, 0.25, na.rm = TRUE)
  Q3 <- quantile(x, 0.75, na.rm = TRUE)
  IQR_val <- Q3 - Q1
  pmin(pmax(x, Q1 - 1.5 * IQR_val), Q3 + 1.5 * IQR_val)
}

flights_sample$ARR_DELAY <- iqr_kirpma(flights_sample$ARR_DELAY)

cat("\nAykırı değer kırpma sonrası ARR_DELAY özeti:\n")
print(summary(flights_sample$ARR_DELAY))

# --- 1.6 Değişken Dönüşümleri -----------------------------------------------

# Modelleme için kullanılacak değişken listesi (VIF ve alan bilgisine göre)
model_degiskenleri <- c("ARR_DELAY", "DISTANCE", "CARRIER_DELAY",
                        "WEATHER_DELAY", "NAS_DELAY", "LATE_AIRCRAFT_DELAY")

flights_model <- flights_sample[, model_degiskenleri]

# Normalizasyon / Standardizasyon (caret preProcess ile center + scale)
onisleme <- preProcess(flights_model[, -1], method = c("center", "scale"))
flights_scaled <- predict(onisleme, flights_model)

cat("\nStandardizasyon sonrası değişken ortalamaları (≈ 0 olmalı):\n")
print(round(colMeans(flights_scaled[, -1]), 4))

# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM II: KEŞİFSEL VERİ ANALİZİ (EDA) VE GÖRSELLEŞTİRME
# ─────────────────────────────────────────────────────────────────────────────

# --- 2.1 Korelasyon Matrisi -------------------------------------------------

korelasyon_matrisi <- cor(flights_model)

corrplot(korelasyon_matrisi,
         method = "circle",
         type   = "upper",
         tl.col = "black",
         tl.srt = 45,
         addCoef.col = "black",
         number.cex  = 0.7,
         title = "Değişkenler Arası Korelasyon Matrisi",
         mar   = c(0, 0, 2, 0))

# --- 2.2 Bağımlı Değişken ile Bağımsız Değişkenler Arası İlişkiler ----------

# DEP_DELAY vs ARR_DELAY
p1 <- ggplot(flights_sample, aes(x = DEP_DELAY, y = ARR_DELAY)) +
  geom_point(color = "darkblue", alpha = 0.3, shape = 20, size = 0.5) +
  geom_smooth(method = "lm", color = "red", size = 1, se = TRUE) +
  labs(title = "Kalkış vs Varış Gecikmesi",
       x = "Kalkış Gecikmesi (dk)", y = "Varış Gecikmesi (dk)") +
  theme_minimal()

# DISTANCE vs ARR_DELAY
p2 <- ggplot(flights_sample, aes(x = DISTANCE, y = ARR_DELAY)) +
  geom_point(color = "darkgreen", alpha = 0.3, shape = 20, size = 0.5) +
  geom_smooth(method = "lm", color = "red", size = 1, se = TRUE) +
  labs(title = "Mesafe vs Varış Gecikmesi",
       x = "Uçuş Mesafesi (mil)", y = "Varış Gecikmesi (dk)") +
  theme_minimal()

# CARRIER_DELAY vs ARR_DELAY
p3 <- ggplot(flights_sample, aes(x = CARRIER_DELAY, y = ARR_DELAY)) +
  geom_point(color = "darkorange", alpha = 0.3, shape = 20, size = 0.5) +
  geom_smooth(method = "lm", color = "red", size = 1, se = TRUE) +
  labs(title = "Havayolu Gecikmesi vs Varış Gecikmesi",
       x = "Havayolu Gecikmesi (dk)", y = "Varış Gecikmesi (dk)") +
  theme_minimal()

# NAS_DELAY vs ARR_DELAY
p4 <- ggplot(flights_sample, aes(x = NAS_DELAY, y = ARR_DELAY)) +
  geom_point(color = "purple", alpha = 0.3, shape = 20, size = 0.5) +
  geom_smooth(method = "lm", color = "red", size = 1, se = TRUE) +
  labs(title = "NAS Gecikmesi vs Varış Gecikmesi",
       x = "NAS Gecikmesi (dk)", y = "Varış Gecikmesi (dk)") +
  theme_minimal()

grid.arrange(p1, p2, p3, p4, ncol = 2,
             top = "Bağımlı Değişken (ARR_DELAY) ile Bağımsız Değişkenler Arası İlişkiler")

# ARR_DELAY dağılımı (histogram)
ggplot(flights_sample, aes(x = ARR_DELAY)) +
  geom_histogram(binwidth = 5, fill = "steelblue", color = "white", alpha = 0.8) +
  labs(title = "Varış Gecikmesi (ARR_DELAY) Dağılımı",
       x = "Varış Gecikmesi (Dakika)", y = "Frekans") +
  theme_minimal()

# --- 2.3 Çoklu Doğrusal Bağlantı (VIF) Kontrolü ----------------------------

gecici_model <- lm(ARR_DELAY ~ DISTANCE + CARRIER_DELAY +
                     WEATHER_DELAY + NAS_DELAY + LATE_AIRCRAFT_DELAY,
                   data = flights_sample)

vif_sonuclari <- vif(gecici_model)
cat("\n========== VIF DEĞERLERİ ==========\n")
print(vif_sonuclari)
cat("Not: VIF < 5 → çoklu bağlantı sorunu yok.\n")
cat("     VIF > 10 → değişken çıkarılmalı veya dönüştürülmeli.\n")

# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM III: MODELLEME STRATEJİLERİ
# ─────────────────────────────────────────────────────────────────────────────

# --- 3.0 Eğitim / Test Bölmesi ve Ortak trainControl ------------------------

# Modelleme için daha küçük bir örneklem kullanıyoruz (hafıza ve süre için)
set.seed(42)
idx          <- createDataPartition(flights_scaled$ARR_DELAY, p = 0.80, list = FALSE)
egitim_seti  <- flights_scaled[ idx, ]
test_seti    <- flights_scaled[-idx, ]

# Tüm modeller aynı 10-katlı Çapraz Doğrulama ile eğitilecek
ctrl <- trainControl(method  = "cv",
                     number  = 10,
                     verboseIter = FALSE,
                     allowParallel = TRUE)

cat("\nEğitim seti boyutu:", nrow(egitim_seti),
    "| Test seti boyutu:", nrow(test_seti), "\n")

# ─── Klasik İstatistiksel Öğrenme Modelleri ──────────────────────────────────

# --- 3.1 Çoklu Doğrusal Regresyon (MLR) ------------------------------------

set.seed(42)
model_lm <- train(ARR_DELAY ~ .,
                  data      = egitim_seti,
                  method    = "lm",
                  trControl = ctrl)

cat("\n--- Çoklu Doğrusal Regresyon Sonuçları ---\n")
print(model_lm)
print(summary(model_lm$finalModel))

# --- 3.2 Ridge / Lasso / Elastic Net (glmnet) --------------------------------
# Hem regularizasyon hem de klasik regresyonun genelleştirilmiş hali

set.seed(42)
model_glmnet <- train(ARR_DELAY ~ .,
                      data      = egitim_seti,
                      method    = "glmnet",
                      trControl = ctrl,
                      tuneLength = 10)  # alpha (0=Ridge, 1=Lasso) ve lambda otomatik aranır

cat("\n--- Elastic Net (glmnet) En İyi Parametreler ---\n")
print(model_glmnet$bestTune)

# ─── Modern Makine Öğrenmesi Modelleri ───────────────────────────────────────

# --- 3.3 Random Forest (rf) --------------------------------------------------

set.seed(42)
model_rf <- train(ARR_DELAY ~ .,
                  data       = egitim_seti,
                  method     = "rf",
                  trControl  = ctrl,
                  tuneLength = 5,        # mtry için 5 farklı değer dener
                  importance = TRUE)

cat("\n--- Random Forest En İyi mtry ---\n")
print(model_rf$bestTune)

# --- 3.4 Gradient Boosting Machine (gbm) ------------------------------------

set.seed(42)
model_gbm <- train(ARR_DELAY ~ .,
                   data      = egitim_seti,
                   method    = "gbm",
                   trControl = ctrl,
                   tuneLength = 5,
                   verbose   = FALSE)

cat("\n--- GBM En İyi Parametreler ---\n")
print(model_gbm$bestTune)

# --- 3.5 k-En Yakın Komşu (knn) ---------------------------------------------

set.seed(42)
model_knn <- train(ARR_DELAY ~ .,
                   data       = egitim_seti,
                   method     = "knn",
                   trControl  = ctrl,
                   tuneGrid   = data.frame(k = c(3, 5, 7, 9, 11, 15)))

cat("\n--- KNN En İyi k Değeri ---\n")
print(model_knn$bestTune)

# --- 3.6 Support Vector Machine - Radial Çekirdek (svmRadial) ---------------

set.seed(42)
model_svm <- train(ARR_DELAY ~ .,
                   data       = egitim_seti,
                   method     = "svmRadial",
                   trControl  = ctrl,
                   tuneLength = 5,
                   preProcess = NULL)   # zaten standardize edildi

cat("\n--- SVM En İyi Parametreler ---\n")
print(model_svm$bestTune)

# --- 3.7 Yapay Sinir Ağı (nnet) ----------------------------------------------

set.seed(42)
model_nnet <- train(ARR_DELAY ~ .,
                    data      = egitim_seti,
                    method    = "nnet",
                    trControl = ctrl,
                    tuneGrid  = expand.grid(size  = c(3, 5, 7),
                                            decay = c(0.001, 0.01, 0.1)),
                    linout    = TRUE,   # Regresyon için zorunlu
                    maxit     = 200,
                    trace     = FALSE)

cat("\n--- Yapay Sinir Ağı En İyi Parametreler ---\n")
print(model_nnet$bestTune)

# Paralel işlemeyi kapat
stopCluster(cl)
registerDoSEQ()

# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM IV: PERFORMANS KARŞILAŞTIRMA VE MODEL SEÇİMİ
# ─────────────────────────────────────────────────────────────────────────────

# --- 4.1 Modellerin resamples() ile Karşılaştırılması -----------------------

model_listesi <- resamples(list(
  "Dogrusal Regresyon" = model_lm,
  "Elastic Net"        = model_glmnet,
  "Random Forest"      = model_rf,
  "GBM"               = model_gbm,
  "KNN"               = model_knn,
  "SVM (Radial)"       = model_svm,
  "Sinir Agi"          = model_nnet
))

cat("\n========== ÇAPRAZ DOĞRULAMA KARŞILAŞTIRMA ÖZETİ ==========\n")
print(summary(model_listesi))

# RMSE karşılaştırma grafiği (kutu grafiği)
bwplot(model_listesi,
       metric = "RMSE",
       main   = "Model Karşılaştırması — RMSE (10-Katlı CV)")

# R-Kare karşılaştırma grafiği (nokta grafiği)
dotplot(model_listesi,
        metric = "Rsquared",
        main   = "Model Karşılaştırması — R² (10-Katlı CV)")

# --- 4.2 Test Seti Performansı (Hata Metrikleri) ----------------------------

# Her model için test seti üzerinde tahmin yap ve metrikleri hesapla
hesapla_metrik <- function(model, test, model_adi) {
  tahmin <- predict(model, newdata = test)
  gercek <- test$ARR_DELAY
  rmse   <- sqrt(mean((gercek - tahmin)^2))
  mae    <- mean(abs(gercek - tahmin))
  r2     <- cor(gercek, tahmin)^2
  data.frame(Model = model_adi, RMSE = round(rmse, 3),
             MAE   = round(mae, 3), R2 = round(r2, 4))
}

test_performans <- rbind(
  hesapla_metrik(model_lm,     test_seti, "Dogrusal Regresyon"),
  hesapla_metrik(model_glmnet, test_seti, "Elastic Net"),
  hesapla_metrik(model_rf,     test_seti, "Random Forest"),
  hesapla_metrik(model_gbm,    test_seti, "GBM"),
  hesapla_metrik(model_knn,    test_seti, "KNN"),
  hesapla_metrik(model_svm,    test_seti, "SVM (Radial)"),
  hesapla_metrik(model_nnet,   test_seti, "Sinir Agi")
)

cat("\n========== TEST SETİ PERFORMANS TABLOSU ==========\n")
print(test_performans[order(test_performans$RMSE), ])

# RMSE bar grafiği (test seti)
ggplot(test_performans, aes(x = reorder(Model, RMSE), y = RMSE, fill = RMSE)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = round(RMSE, 2)), hjust = -0.1, size = 3.5) +
  coord_flip() +
  scale_fill_gradient(low = "steelblue", high = "firebrick") +
  labs(title = "Test Seti RMSE Karşılaştırması",
       x = NULL, y = "RMSE (Dakika)") +
  theme_minimal() +
  theme(legend.position = "none")

# --- 4.3 Değişken Önem Düzeyleri (varImp) ------------------------------------

# En iyi performanslı model için değişken önem grafiği

en_iyi_model <- model_nnet

cat("\n========== DEĞİŞKEN ÖNEM DÜZEYLERİ (Random Forest) ==========\n")
varimportance <- varImp(en_iyi_model, scale = TRUE)
print(varimportance)
plot(varimportance,
     main = "Değişken Önem Düzeyleri (Random Forest)",
     col  = "steelblue")

# --- 4.4 İndirgenmiş (Final) Model ------------------------------------------
# varImp sonucuna göre en önemli 3 değişken seçilip sadeleştirilmiş model kurulur

set.seed(42)
model_final <- train(ARR_DELAY ~ CARRIER_DELAY + NAS_DELAY + LATE_AIRCRAFT_DELAY,
                     data      = egitim_seti,
                     method    = "lm",
                     trControl = ctrl)

cat("\n========== İNDİRGENMİŞ FİNAL MODEL ==========\n")
print(summary(model_final$finalModel))

# Final modelin test seti performansı
final_tahmin <- predict(model_final, newdata = test_seti)
final_rmse   <- sqrt(mean((test_seti$ARR_DELAY - final_tahmin)^2))
final_r2     <- cor(test_seti$ARR_DELAY, final_tahmin)^2
cat(sprintf("Final Model → RMSE: %.3f | R²: %.4f\n", final_rmse, final_r2))

# Gerçek vs Tahmin grafiği (final model)
sonuc_df <- data.frame(Gercek  = test_seti$ARR_DELAY,
                       Tahmin  = final_tahmin)

ggplot(sonuc_df[sample(nrow(sonuc_df), 1000), ], aes(x = Gercek, y = Tahmin)) +
  geom_point(color = "steelblue", alpha = 0.4, shape = 20) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", size = 1) +
  labs(title = "Final Model: Gerçek vs Tahmin Edilen Varış Gecikmesi",
       x = "Gerçek ARR_DELAY (dk)",
       y = "Tahmin Edilen ARR_DELAY (dk)") +
  theme_minimal()

# =============================================================================
# ÖZET NOTLAR
# Bağımlı değişken   : ARR_DELAY (sürekli → Regresyon problemi)
# Karşılaştırma metrikleri: RMSE, MAE, R²
# Çapraz doğrulama   : 10-katlı (tüm modellerde aynı ctrl yapısı)
# Hiperparametre opt.: tuneLength / tuneGrid ile otomatik
# =============================================================================
