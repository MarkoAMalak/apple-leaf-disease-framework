# رفع المستودع على GitHub + Zenodo (خطوات تعملها إنت)

> ملاحظة: أنا جهّزت المستودع كامل وجاهز، بس **الرفع لازم تعمله إنت** لأنه محتاج تسجيل دخول بحسابك.

## أولًا: إنشاء المستودع على GitHub

1. من [github.com/new](https://github.com/new) اعمل repo جديد باسم مثلاً `apple-leaf-disease-framework`.
2. خلّيه **Public** (عشان Zenodo يقدر يأرشفه ويطلّع DOI)، ومتحطّش README/‏.gitignore (عندنا بتوعنا).

## ثانيًا: الرفع من جهازك

افتح terminal جوّه مجلد `apple-leaf-disease-framework` (بعد ما تفكّ الضغط) ونفّذ:

```bash
git init
git add .
git status          # اتأكد إن مفيش أي صور/داتا/أوزان (.jpg .png خام أو .pt) اتضافت
git commit -m "Leakage-free apple-leaf disease framework: code, notebooks, results, paper, thesis"
git branch -M main
git remote add origin https://github.com/MarkoAMalak/apple-leaf-disease-framework.git
git push -u origin main
```

> ملف `.gitignore` بيمنع تلقائيًا رفع أي داتا أو صور خام أو أوزان (`.pt`). لو `git status`
> أظهر أي ملف داتا، **متعملش commit** وقوللي.

## ثالثًا: Zenodo DOI (أرشفة + رقم تعريف دائم)

1. ادخل [zenodo.org](https://zenodo.org/) وسجّل دخول **بحساب GitHub** بتاعك.
2. من الإعدادات → **GitHub**: [zenodo.org/account/settings/github](https://zenodo.org/account/settings/github/)
3. فعّل السويتش جنب مستودع `apple-leaf-disease-framework` (Flip it ON).
4. ارجع GitHub → في المستودع اعمل **Release** جديد (Releases → Draft a new release → tag `v1.0.0` → Publish).
5. Zenodo هيلتقط الـ release تلقائيًا ويطلّعلك **DOI**. انسخه.

## رابعًا: حطّ الـ DOI في الورقة والرسالة

ابعتلي الـ DOI وأنا أحطّه مكان الـ placeholder في:
- بيان "Data Availability" في الورقة.
- قسم Reproducibility في الرسالة.
- وممكن أضيف badge للـ DOI في الـ README.

---

### اللي جوّه المستودع
- `notebooks/` — كل النوتبوكس بآخر وضع (التصنيف، التقييم الحقلي، الكشف، كفاءة البيانات).
- `scripts/` — نسخ سطر أوامر من التجارب.
- `results/` — ملخصات النتائج فقط (CSV + منحنيات PNG) — **مفيش داتا خام**.
- `paper/` و `thesis/` — الورقة والرسالة PDF (+ DOCX للرسالة).
- `docs/RUNNING.md` — كل الأوامر.
