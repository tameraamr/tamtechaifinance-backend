# 📅 Weekly Ticker Refresh Guide

## 🎯 Overview

All 270 stock tickers are now cached for **7 days** instead of 24 hours. To keep them fresh, run a weekly refresh.

**Cost**: $0.54 per refresh (270 tickers × $0.002)
**Time**: ~18 minutes (270 tickers × 4 seconds)

---

## 🚀 How to Refresh All Tickers

### **Option 1: Using Postman or Browser**

```
POST https://tamtechaifinance-backend-production.up.railway.app/admin/refresh-all-tickers?admin_key=tamtech_refresh_2026
```

**Response:**
```json
{
  "success": true,
  "message": "Refresh started in background",
  "total_tickers": 270,
  "stale_reports": 45,
  "estimated_time": "18 minutes",
  "estimated_cost": "$0.54",
  "status": "Check server logs for progress"
}
```

### **Option 2: Using PowerShell**

```powershell
Invoke-WebRequest -Uri "https://tamtechaifinance-backend-production.up.railway.app/admin/refresh-all-tickers?admin_key=tamtech_refresh_2026" -Method POST
```

### **Option 3: Using curl**

```bash
curl -X POST "https://tamtechaifinance-backend-production.up.railway.app/admin/refresh-all-tickers?admin_key=tamtech_refresh_2026"
```

---

## 📊 Check Cache Status

Before refreshing, check which tickers need updating:

```
GET https://tamtechaifinance-backend-production.up.railway.app/admin/cache-status
```

**Response:**
```json
{
  "total_reports": 270,
  "fresh_24h": 50,
  "fresh_7d": 225,
  "stale_7d_plus": 45,
  "coverage": "100%",
  "total_tickers_in_pool": 270
}
```

---

## 🔒 Security

**Default Admin Key**: `tamtech_refresh_2026`

**To Change**:
1. Set environment variable in Railway: `ADMIN_REFRESH_KEY=your_secret_key`
2. Use your new key in the URL: `?admin_key=your_secret_key`

---

## ⏰ Recommended Schedule

**Every Monday at 9 AM**:
1. Check `/admin/cache-status` to see what's stale
2. Run `/admin/refresh-all-tickers`
3. Wait 18 minutes (runs in background)
4. Check Railway logs to confirm success

---

## 📈 What Happens During Refresh

1. **Skips Fresh Reports**: Already < 7 days old
2. **Updates Stale Reports**: > 7 days old
3. **Respects Rate Limits**: 4 seconds between requests (15 RPM)
4. **Circuit Breaker Protection**: Stops if Gemini API fails
5. **Background Task**: Returns immediately, runs in background

---

## 🎯 Benefits of 7-Day Cache

✅ **85% Cost Reduction**: $70/month → $10/month  
✅ **Better Performance**: More cache hits = faster responses  
✅ **SEO Optimized**: All 270 pages stay indexed  
✅ **Rate Limit Safe**: Fewer API calls = less throttling  
✅ **Scalable**: Can handle 10x more traffic  

---

## 🛠️ Monitoring

**Check Railway Logs** to see progress:
```
🔄 Refreshing AAPL... (1/270)
  ✅ AAPL refreshed successfully
🔄 Refreshing MSFT... (2/270)
  ✓ MSFT already fresh (2 days old)
...
🎉 BATCH REFRESH COMPLETE:
  ✅ Refreshed: 180
  ⏭️  Skipped (already fresh): 85
  ✗ Failed: 5
  📊 Total: 270
```

---

## ❓ FAQ

**Q: What if some tickers fail?**  
A: They'll be retried next week. Check logs to see which failed.

**Q: Can I refresh during business hours?**  
A: Yes! It runs in background and won't affect users.

**Q: What if I forget to refresh?**  
A: Old reports stay available for SEO. Users can still manually refresh.

**Q: Can I refresh specific tickers only?**  
A: Not yet, but you can use the regular `/analyze/{ticker}?force_refresh=true` endpoint.

---

**Created**: February 2, 2026  
**Last Updated**: February 2, 2026
