import React, { useState, useEffect } from 'react';
import { AlertTriangle, ThermometerSun, Droplets, Zap, TrendingUp, CheckCircle, XCircle, AlertCircle, Activity, MapPin } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';

const VaxGuardDashboard = () => {
  const [fridges, setFridges] = useState([]);
  const [selectedFridge, setSelectedFridge] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [timeSeriesData, setTimeSeriesData] = useState([]);
  const [globalStats, setGlobalStats] = useState({
    totalUnits: 12,
    atRisk: 2,
    healthy: 9,
    critical: 1
  });

  // Initialize fridges
  useEffect(() => {
    const initialFridges = Array.from({ length: 12 }, (_, i) => ({
      id: `FRIDGE-${String(i + 1).padStart(3, '0')}`,
      location: ['District Hospital A', 'Rural Clinic B', 'Vaccination Center C', 'Mobile Unit D', 'Health Post E', 'Regional Storage F', 'Transit Hub G', 'Emergency Depot H', 'Community Center I', 'Field Station J', 'Border Clinic K', 'Remote Outpost L'][i],
      temperature: 2 + Math.random() * 4,
      humidity: 40 + Math.random() * 20,
      powerStatus: Math.random() > 0.1,
      riskScore: Math.random() * 30,
      vaccineType: ['COVID-19', 'Measles', 'Polio', 'HPV'][Math.floor(Math.random() * 4)],
      doses: Math.floor(Math.random() * 5000) + 1000,
      lastSync: new Date(Date.now() - Math.random() * 300000)
    }));
    setFridges(initialFridges);
    setSelectedFridge(initialFridges[0]);
  }, []);

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setFridges(prev => prev.map(fridge => {
        let newTemp = fridge.temperature + (Math.random() - 0.5) * 0.5;
        let newRisk = fridge.riskScore;

        // Simulate breach scenario for fridge 7
        if (fridge.id === 'FRIDGE-007') {
          newTemp = Math.min(newTemp + 0.15, 9);
          newRisk = Math.min(((newTemp - 2) / 6) * 100, 95);
        } else {
          newTemp = Math.max(2, Math.min(8, newTemp));
          newRisk = Math.max(0, newRisk + (Math.random() - 0.7) * 5);
        }

        return {
          ...fridge,
          temperature: newTemp,
          humidity: Math.max(35, Math.min(65, fridge.humidity + (Math.random() - 0.5) * 2)),
          riskScore: newRisk,
          lastSync: new Date()
        };
      }));
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  // Generate time series data
  useEffect(() => {
    if (!selectedFridge) return;

    const interval = setInterval(() => {
      setTimeSeriesData(prev => {
        const newData = [...prev, {
          time: new Date().toLocaleTimeString(),
          temperature: selectedFridge.temperature,
          riskScore: selectedFridge.riskScore,
          threshold: 8
        }].slice(-20);
        return newData;
      });
    }, 2000);

    return () => clearInterval(interval);
  }, [selectedFridge]);

  // Update alerts
  useEffect(() => {
    const newAlerts = fridges
      .filter(f => f.riskScore > 50)
      .map(f => ({
        id: f.id,
        severity: f.riskScore > 75 ? 'critical' : 'warning',
        message: `${f.id} - Risk score: ${f.riskScore.toFixed(1)}% | ${f.location}`,
        time: new Date().toLocaleTimeString()
      }))
      .slice(0, 5);
    
    setAlerts(newAlerts);

    setGlobalStats({
      totalUnits: fridges.length,
      critical: fridges.filter(f => f.riskScore > 75).length,
      atRisk: fridges.filter(f => f.riskScore > 50 && f.riskScore <= 75).length,
      healthy: fridges.filter(f => f.riskScore <= 50).length
    });
  }, [fridges]);

  const getRiskColor = (score) => {
    if (score > 75) return 'text-red-600 bg-red-50 border-red-300';
    if (score > 50) return 'text-orange-600 bg-orange-50 border-orange-300';
    if (score > 25) return 'text-yellow-600 bg-yellow-50 border-yellow-300';
    return 'text-green-600 bg-green-50 border-green-300';
  };

  const getRiskBadge = (score) => {
    if (score > 75) return { text: 'CRITICAL', color: 'bg-red-600' };
    if (score > 50) return { text: 'HIGH RISK', color: 'bg-orange-500' };
    if (score > 25) return { text: 'MONITOR', color: 'bg-yellow-500' };
    return { text: 'SAFE', color: 'bg-green-500' };
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-cyan-50 p-4">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <div>
            <h1 className="text-4xl font-bold text-gray-900 flex items-center gap-3">
              <Activity className="text-blue-600" size={40} />
              VaxGuard AI
            </h1>
            <p className="text-gray-600 mt-1">Global Vaccine Cold-Chain Intelligence System</p>
          </div>
          <div className="text-right text-sm text-gray-500">
            <div>Last Updated: {new Date().toLocaleTimeString()}</div>
            <div className="text-xs mt-1">All systems operational</div>
          </div>
        </div>
      </div>

      {/* Global Stats */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <div className="bg-white p-4 rounded-lg shadow-md border-l-4 border-blue-500">
          <div className="text-sm text-gray-600 mb-1">Total Units</div>
          <div className="text-3xl font-bold text-gray-900">{globalStats.totalUnits}</div>
        </div>
        <div className="bg-white p-4 rounded-lg shadow-md border-l-4 border-green-500">
          <div className="text-sm text-gray-600 mb-1 flex items-center gap-1">
            <CheckCircle size={14} /> Healthy
          </div>
          <div className="text-3xl font-bold text-green-600">{globalStats.healthy}</div>
        </div>
        <div className="bg-white p-4 rounded-lg shadow-md border-l-4 border-orange-500">
          <div className="text-sm text-gray-600 mb-1 flex items-center gap-1">
            <AlertCircle size={14} /> At Risk
          </div>
          <div className="text-3xl font-bold text-orange-600">{globalStats.atRisk}</div>
        </div>
        <div className="bg-white p-4 rounded-lg shadow-md border-l-4 border-red-500">
          <div className="text-sm text-gray-600 mb-1 flex items-center gap-1">
            <XCircle size={14} /> Critical
          </div>
          <div className="text-3xl font-bold text-red-600">{globalStats.critical}</div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-6">
        {/* Left Column - Fridge List */}
        <div className="col-span-1 space-y-4">
          <div className="bg-white rounded-lg shadow-md p-4">
            <h2 className="text-lg font-bold text-gray-900 mb-3 flex items-center gap-2">
              <MapPin size={20} className="text-blue-600" />
              Cold Storage Units
            </h2>
            <div className="space-y-2 max-h-[600px] overflow-y-auto">
              {fridges.sort((a, b) => b.riskScore - a.riskScore).map(fridge => {
                const badge = getRiskBadge(fridge.riskScore);
                return (
                  <div
                    key={fridge.id}
                    onClick={() => setSelectedFridge(fridge)}
                    className={`p-3 rounded-lg border-2 cursor-pointer transition-all ${
                      selectedFridge?.id === fridge.id 
                        ? 'border-blue-500 bg-blue-50' 
                        : getRiskColor(fridge.riskScore)
                    }`}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="font-semibold text-sm">{fridge.id}</div>
                      <span className={`text-xs px-2 py-1 rounded-full text-white ${badge.color}`}>
                        {badge.text}
                      </span>
                    </div>
                    <div className="text-xs text-gray-600 mb-2">{fridge.location}</div>
                    <div className="flex items-center gap-3 text-xs">
                      <div className="flex items-center gap-1">
                        <ThermometerSun size={12} />
                        {fridge.temperature.toFixed(1)}°C
                      </div>
                      <div className="flex items-center gap-1">
                        <TrendingUp size={12} />
                        {fridge.riskScore.toFixed(0)}%
                      </div>
                      {!fridge.powerStatus && <Zap size={12} className="text-red-500" />}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Middle Column - Detailed View */}
        <div className="col-span-2 space-y-4">
          {selectedFridge && (
            <>
              {/* Unit Details */}
              <div className="bg-white rounded-lg shadow-md p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-2xl font-bold text-gray-900">{selectedFridge.id}</h2>
                  <span className={`px-4 py-2 rounded-full text-white text-sm font-bold ${getRiskBadge(selectedFridge.riskScore).color}`}>
                    RISK: {selectedFridge.riskScore.toFixed(1)}%
                  </span>
                </div>

                <div className="grid grid-cols-3 gap-4 mb-4">
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <div className="flex items-center gap-2 text-gray-600 text-sm mb-1">
                      <ThermometerSun size={16} />
                      Temperature
                    </div>
                    <div className="text-2xl font-bold text-gray-900">
                      {selectedFridge.temperature.toFixed(2)}°C
                    </div>
                    <div className="text-xs text-gray-500 mt-1">Target: 2-8°C</div>
                  </div>

                  <div className="bg-gray-50 p-4 rounded-lg">
                    <div className="flex items-center gap-2 text-gray-600 text-sm mb-1">
                      <Droplets size={16} />
                      Humidity
                    </div>
                    <div className="text-2xl font-bold text-gray-900">
                      {selectedFridge.humidity.toFixed(1)}%
                    </div>
                    <div className="text-xs text-gray-500 mt-1">Normal: 40-60%</div>
                  </div>

                  <div className="bg-gray-50 p-4 rounded-lg">
                    <div className="flex items-center gap-2 text-gray-600 text-sm mb-1">
                      <Zap size={16} />
                      Power Status
                    </div>
                    <div className="text-2xl font-bold text-gray-900">
                      {selectedFridge.powerStatus ? 'ON' : 'BACKUP'}
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      {selectedFridge.powerStatus ? 'Mains Active' : 'Battery Mode'}
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-600">Location:</span>
                    <span className="ml-2 font-semibold">{selectedFridge.location}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Vaccine Type:</span>
                    <span className="ml-2 font-semibold">{selectedFridge.vaccineType}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Doses Stored:</span>
                    <span className="ml-2 font-semibold">{selectedFridge.doses.toLocaleString()}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Last Sync:</span>
                    <span className="ml-2 font-semibold">{selectedFridge.lastSync.toLocaleTimeString()}</span>
                  </div>
                </div>
              </div>

              {/* Time Series Chart */}
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-bold text-gray-900 mb-4">Real-Time Monitoring</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={timeSeriesData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" tick={{ fontSize: 11 }} />
                    <YAxis yAxisId="left" domain={[0, 10]} tick={{ fontSize: 11 }} />
                    <YAxis yAxisId="right" orientation="right" domain={[0, 100]} tick={{ fontSize: 11 }} />
                    <Tooltip />
                    <Legend />
                    <Line yAxisId="left" type="monotone" dataKey="temperature" stroke="#3b82f6" strokeWidth={2} name="Temp (°C)" />
                    <Line yAxisId="left" type="monotone" dataKey="threshold" stroke="#dc2626" strokeWidth={2} strokeDasharray="5 5" name="Threshold" />
                    <Line yAxisId="right" type="monotone" dataKey="riskScore" stroke="#f59e0b" strokeWidth={2} name="Risk (%)" />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* AI Recommendations */}
              {selectedFridge.riskScore > 50 && (
                <div className={`rounded-lg p-4 ${selectedFridge.riskScore > 75 ? 'bg-red-50 border-2 border-red-300' : 'bg-orange-50 border-2 border-orange-300'}`}>
                  <div className="flex items-start gap-3">
                    <AlertTriangle className={selectedFridge.riskScore > 75 ? 'text-red-600' : 'text-orange-600'} size={24} />
                    <div>
                      <h4 className="font-bold text-gray-900 mb-2">AI-Recommended Actions</h4>
                      <ul className="text-sm space-y-1 text-gray-700">
                        {selectedFridge.riskScore > 75 && (
                          <>
                            <li>• URGENT: Inspect cooling system immediately</li>
                            <li>• Transfer vaccines to backup storage within 30 minutes</li>
                            <li>• Contact maintenance team</li>
                          </>
                        )}
                        {selectedFridge.riskScore > 50 && selectedFridge.riskScore <= 75 && (
                          <>
                            <li>• Schedule maintenance check within 2 hours</li>
                            <li>• Monitor temperature every 15 minutes</li>
                            <li>• Prepare backup storage unit</li>
                          </>
                        )}
                      </ul>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* Active Alerts Footer */}
      {alerts.length > 0 && (
        <div className="mt-6 bg-white rounded-lg shadow-md p-4">
          <h3 className="text-lg font-bold text-gray-900 mb-3 flex items-center gap-2">
            <AlertTriangle className="text-orange-600" size={20} />
            Active Alerts ({alerts.length})
          </h3>
          <div className="space-y-2">
            {alerts.map((alert, idx) => (
              <div key={idx} className={`flex items-center justify-between p-3 rounded-lg ${alert.severity === 'critical' ? 'bg-red-50 border-l-4 border-red-500' : 'bg-orange-50 border-l-4 border-orange-500'}`}>
                <div className="flex items-center gap-3">
                  {alert.severity === 'critical' ? <XCircle className="text-red-600" size={18} /> : <AlertCircle className="text-orange-600" size={18} />}
                  <span className="text-sm font-medium text-gray-900">{alert.message}</span>
                </div>
                <span className="text-xs text-gray-500">{alert.time}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default VaxGuardDashboard;
