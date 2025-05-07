# CarSense Backend Architecture & ML Service Integration Guide

## Project Overview

CarSense is a comprehensive vehicle diagnostics ecosystem designed for the Romanian automotive market, consisting of three main components:

1. **Android OBD2 Mobile App**: Connects to vehicle OBD2 ports to collect diagnostic data
2. **API Backend** (this repository): Stores and processes vehicle diagnostic data
3. **ML Service** (separate project): Provides predictive maintenance and diagnostic recommendations

The system aims to help vehicle owners identify potential problems before they lead to breakdowns, optimize maintenance schedules, and reduce repair costs through early detection.

## Business Context

The Romanian automotive market has several specific characteristics that CarSense addresses:
- High proportion of older vehicles (average age ~16 years)
- Predominance of specific makes: Dacia, Volkswagen, Renault, Ford, and Skoda
- Varying road conditions that affect vehicle wear patterns
- Need for affordable preventive maintenance solutions

CarSense specializes in providing diagnostic services that understand these market dynamics and deliver targeted, actionable maintenance recommendations to Romanian drivers.

## Backend Architecture Overview

The CarSense backend is a TypeScript-based API service built with the following core technologies:

- **Hono**: Web framework for API routing and middleware
- **PostgreSQL**: Relational database for persistent storage
- **Drizzle ORM**: Type-safe database ORM layer
- **Zod**: Schema validation and type generation
- **Bun**: JavaScript/TypeScript runtime (preferred over Node.js)
- **OpenAPI**: API documentation generation

### Directory Structure

```
carsense-backend
├── .vscode/              # VS Code configuration
├── migrations/           # Database migration files
│   └── meta/             # Migration metadata
├── src/                  # Source code
│   ├── db/               # Database configuration
│   │   └── schema/       # Drizzle schema definitions
│   ├── lib/              # Utility functions and shared code
│   ├── middleware/       # API middleware (auth, logging, etc.)
│   ├── routes/           # API route handlers
│   │   └── api/          # API endpoints
│   ├── scripts/          # Utility scripts
│   │   ├── convert-dtc.ts      # Script to convert DTC codes to JSON
│   │   └── import-dtc-codes.ts # Script to import DTC codes to database
│   └── zod/              # Zod schemas for validation
├── .env                  # Environment variables
├── drizzle.config.ts     # Drizzle ORM configuration
├── dtc.bin               # Binary DTC data file (source format)
├── dtc.json              # JSON-formatted DTC data (processed format)
├── env.ts                # Environment variable type definitions
├── AUTOMOTIVE_DATASETS_AND_RESOURCES.md  # ML training data resources
├── ML_SERVICE_ROADMAP.md # Roadmap for ML service implementation
└── package.json          # Project dependencies
```

### API Endpoints

The backend exposes several REST API endpoints, including:

- `/api/dtc?code=P0001`: Retrieve diagnostic trouble code information
  - Returns detailed information about specific DTC codes
  - Supports query parameter filtering
- `/api/vehicles`: Vehicle management endpoints
  - GET: Retrieve vehicles registered to a user
  - POST: Register a new vehicle
- `/api/readings`: OBD2 reading storage and retrieval
  - POST: Store new readings from the mobile app
  - GET: Retrieve historical readings for a vehicle
- `/api/users`: User management and authentication
  - POST: Register new users
  - Authentication endpoints for login/logout

### Database Schema

The database includes tables for:
- Diagnostic Trouble Codes (DTCs)
  - Code (e.g., P0001)
  - Description
  - Possible causes
  - Severity
  - Components affected

## Database Architecture

### Technology Stack

The CarSense backend uses PostgreSQL as its primary database, with Drizzle ORM as the data access layer. This combination provides:

- **Type Safety**: Full TypeScript type integration with the database schema
- **Migration Management**: Robust schema versioning through Drizzle's migration system
- **Performance**: Optimized query building and execution
- **Scalability**: PostgreSQL's proven scalability for growing datasets

### Schema Design

The database schema is defined in TypeScript using Drizzle's schema definition language in the `src/db/schema/` directory. The main tables include:

#### 1. Diagnostic Trouble Codes (`dtc`)
```typescript
// Simplified representation of the DTC table schema
export const dtc = pgTable('dtc', {
  id: serial('id').primaryKey(),
  code: text('code').notNull().unique(),
  description: text('description').notNull(),
  possibleCauses: text('possible_causes'),
  severity: text('severity'),
  components: text('components'),
  created_at: timestamp('created_at').defaultNow(),
  updated_at: timestamp('updated_at').defaultNow()
});
```

#### 2. Vehicles
Stores information about specific vehicles registered in the system:
```typescript
// Simplified representation
export const vehicles = pgTable('vehicles', {
  id: serial('id').primaryKey(),
  make: text('make').notNull(),
  model: text('model').notNull(),
  year: integer('year').notNull(),
  engineType: text('engine_type'),
  vin: text('vin').unique(),
  userId: integer('user_id').references(() => users.id),
  created_at: timestamp('created_at').defaultNow(),
  updated_at: timestamp('updated_at').defaultNow()
});
```

#### 3. OBD Readings
Stores the OBD parameter readings collected from vehicles:
```typescript
// Simplified representation
export const obdReadings = pgTable('obd_readings', {
  id: serial('id').primaryKey(),
  vehicleId: integer('vehicle_id').references(() => vehicles.id),
  timestamp: timestamp('timestamp').defaultNow(),
  parameters: jsonb('parameters'), // Stores all OBD parameters as JSON
  dtcCodes: text('dtc_codes').array(),
  location: jsonb('location').default({})
});
```

#### 4. Users
Stores user information for authentication and profile management:
```typescript
// Simplified representation
export const users = pgTable('users', {
  id: serial('id').primaryKey(),
  email: text('email').notNull().unique(),
  passwordHash: text('password_hash').notNull(),
  name: text('name'),
  created_at: timestamp('created_at').defaultNow(),
  updated_at: timestamp('updated_at').defaultNow()
});
```

#### 5. Maintenance Records
Tracks vehicle maintenance events:
```typescript
// Simplified representation
export const maintenanceRecords = pgTable('maintenance_records', {
  id: serial('id').primaryKey(),
  vehicleId: integer('vehicle_id').references(() => vehicles.id),
  serviceDate: timestamp('service_date').notNull(),
  description: text('description').notNull(),
  mileage: integer('mileage'),
  serviceType: text('service_type'),
  cost: numeric('cost'),
  shopName: text('shop_name'),
  created_at: timestamp('created_at').defaultNow()
});
```

### Database Relationships

The database is designed with the following key relationships:

1. **One-to-Many**: A user can have multiple vehicles
2. **One-to-Many**: A vehicle can have multiple OBD readings
3. **One-to-Many**: A vehicle can have multiple maintenance records
4. **Many-to-Many**: Vehicles can have multiple DTCs (through the OBD readings)

### Migration Strategy

Database migrations are managed through Drizzle's migration toolkit, which generates SQL migration files based on schema changes. These are stored in the `migrations/` directory and tracked in version control.

The migration process follows these steps:

1. Update the schema definition in the TypeScript files
2. Generate a migration file with `bun drizzle-kit generate:pg`
3. Apply migrations to the database with `bun migrate`

This approach ensures:
- Version-controlled schema changes
- Safe production deployments
- Ability to roll back changes if needed

### Query Patterns

The backend uses the following query patterns for database access:

1. **Repository Pattern**: Encapsulating database operations in repository classes
2. **Query Building**: Using Drizzle's query builder for complex queries
3. **Transactions**: Ensuring data consistency for multi-table operations

Example query implementation:
```typescript
// Example of a repository method to fetch DTCs by code
export async function getDtcByCode(code: string) {
  return await db.query.dtc.findFirst({
    where: eq(dtc.code, code)
  });
}
```

### Database Performance Optimizations

The database is optimized for performance through:

1. **Indexing**: Strategic indexes on frequently queried columns
2. **JSON Storage**: Using PostgreSQL's JSONB type for flexible storage of OBD parameters
3. **Connection Pooling**: Efficient connection management for API requests
4. **Query Optimization**: Careful design of queries to minimize execution time

### Key Implementation Details

1. **DTC Data Processing**:
   - `convert-dtc.ts`: Script that transforms C# DTC code file into a clean JSON format
   - `import-dtc-codes.ts`: Script that imports the JSON DTC data into the PostgreSQL database

2. **OpenAPI and Zod Integration**:
   - Zod schemas are used for validation and OpenAPI documentation
   - The necessary `"zod-openapi/extend"` import is added to schema files

3. **Environment Configuration**:
   - Environment variables are typed through `env.ts`
   - Drizzle ORM configuration in `drizzle.config.ts`

## ML Service Integration Plan

### Overview and Purpose

The Machine Learning (ML) service will be developed as a separate project that communicates with the main backend. Its primary purposes are:

1. **Predictive Maintenance**: Anticipate potential vehicle failures before they occur
2. **Component Health Monitoring**: Track degradation of specific vehicle components
3. **Maintenance Recommendations**: Suggest optimal timing for maintenance activities
4. **Cost Estimation**: Provide approximate repair costs for detected issues

The ML service will consume OBD2 data and DTC codes from the backend, process them through trained models, and return actionable insights.

### Communication Interface

For integrating the ML service with this backend, the recommended approach is:

1. **REST API Interface**:
   - The ML service should expose endpoints for:
     - Prediction requests based on DTC codes and OBD parameters
     - Model information and metadata
     - Training status and metrics

2. **API Contract Example**:

#### Backend → ML Service Requests
```typescript
// Request to get prediction based on OBD2 parameters
interface PredictionRequest {
  vehicleInfo: {
    make: string;
    model: string;
    year: number;
    engineType: string;
    mileage: number;
  };
  dtcCodes: string[];  // e.g. ["P0101", "P0234"]
  obdParameters: Record<string, number>;  // e.g. {"coolant_temp_c": 95, "engine_load_percent": 78}
}
```

#### ML Service → Backend Responses
```typescript
// Response with prediction results
interface PredictionResponse {
  predictions: {
    componentFailureProbabilities: Record<string, number>;  // e.g. {"Mass Airflow Sensor": 0.89}
    recommendedActions: string[];  // e.g. ["Replace Mass Airflow Sensor", "Check intake system for leaks"]
    urgency: "low" | "medium" | "high" | "critical";
    estimatedRepairCost: {
      min: number;
      max: number;
      currency: string;
    };
    confidence: number;  // 0-1 value indicating model confidence
  };
  modelInfo: {
    version: string;
    lastTrainingDate: string;
    accuracyMetrics: Record<string, number>;
  }
}
```

### Authentication and Security

For secure communication between the backend and ML service:

1. **API Key Authentication**: The backend will include an API key in requests to the ML service
2. **Rate Limiting**: Prevent abuse through appropriate rate limits
3. **Input Validation**: Both services should validate inputs to prevent injection attacks
4. **HTTPS**: All communication should be encrypted

### Deployment Architecture

The recommended deployment setup is:

1. **Containerization**: Both the backend and ML service should be containerized with Docker
2. **Orchestration**: Kubernetes for managing containers and scaling
3. **Service Mesh**: Optional but recommended for secure service-to-service communication
4. **API Gateway**: For routing requests and applying consistent security policies

## Data Resources for ML Model Training

For training the ML service models, refer to the datasets in `AUTOMOTIVE_DATASETS_AND_RESOURCES.md`, which includes:

1. **Public OBD2 Datasets**:
   - KIT Automotive OBD-II Dataset
   - LEVIN Vehicle Telematics Data
   - Hayatu's Automotive Diagnostics Dataset
   - ArjanKw's OBD2 Open Data

2. **European Vehicle-Specific Resources**:
   - BMW Deep OBD Configurations
   - Mercedes-Benz ODX Tools
   - Dacia/Renault Diagnostic Information

3. **Labeled Datasets for Predictive Maintenance**:
   - NHTSA's Office of Defects Investigation Complaints
   - AI4I 2020 Predictive Maintenance Dataset
   - PHM Society Data Repository

4. **Custom Dataset Creation Strategy**:
   - Service center collaboration for repair outcome data
   - DTC-to-repair correlation datasets
   - Vehicle-specific OBD2 data collection

### ML Model Types

The ML service should implement multiple model types:

1. **Classification Models**: For categorizing issues by severity and urgency
2. **Regression Models**: For predicting time-to-failure and repair costs
3. **Clustering Models**: For identifying patterns in vehicle conditions
4. **Ensemble Methods**: For combining multiple models for robust predictions

### Romanian-Specific Considerations

The ML models should account for specific characteristics of the Romanian market:

1. **Vehicle Distribution**: Focus on popular makes/models in Romania (Dacia, VW, Renault)
2. **Age Distribution**: Models should handle older vehicles common in Romania
3. **Seasonal Factors**: Incorporate Romania's distinct seasonal variations
4. **Road Conditions**: Adapt to varying road quality across Romania
5. **Driving Patterns**: Account for specific driving behaviors in Romanian traffic

## Ecosystem Context

The CarSense backend is part of a larger ecosystem:

1. **Android OBD2 App**:
   - Connects to vehicles via OBD2 port
   - Collects real-time diagnostic data
   - Sends data to the backend
   - Features:
     - Real-time vehicle monitoring
     - DTC code reading and clearing
     - Trip tracking and fuel economy monitoring
     - User-friendly diagnostic information display

2. **ML Neural Network**:
   - Predicts potential issues based on OBD2 data and DTC codes
   - Provides maintenance recommendations
   - Being developed as a separate service
   - Features:
     - Multiple specialized models for different vehicle makes
     - Deep learning architectures for temporal data analysis
     - Probabilistic predictions with confidence scores
     - Romanian-specific adaptations

3. **Web Interface**:
   - Dashboard for viewing vehicle health
   - Displays predicted issues and recommendations
   - Administrative tools for fleet management
   - Features:
     - Vehicle health scoring
     - Maintenance tracking
     - Historical data visualization
     - Service center recommendations

### Data Flow Between Components

1. **Mobile App → Backend**:
   - OBD2 parameter readings collected at regular intervals
   - DTC codes when present
   - Trip information and location data
   - User interactions and feedback

2. **Backend → ML Service**:
   - Vehicle metadata
   - Historical OBD2 readings
   - DTC code history
   - Maintenance records when available

3. **ML Service → Backend**:
   - Predictive maintenance recommendations
   - Component health scores
   - Failure probability estimates
   - Suggested repair actions

4. **Backend → Mobile App and Web Interface**:
   - Processed diagnostic information
   - ML-based predictions and recommendations
   - Historical trends and analysis
   - User-specific alerts and notifications

## Development Practices

1. **Schema Documentation**:
   - All Zod schema files should contain structured comments
   - OpenAPI documentation should be generated from schemas

2. **Environment Consistency**:
   - Use Bun for all runtime operations
   - Avoid mixing with Node.js tools

3. **Version Control**:
   - Git-based workflow
   - Feature branches for new functionality
   - Pull requests for code review

4. **Testing Strategy**:
   - Unit tests for core functionality
   - Integration tests for API endpoints
   - Load testing for performance-critical paths
   - Test coverage tracking

5. **Coding Standards**:
   - ESLint for code style enforcement
   - Prettier for consistent formatting
   - TypeScript strict mode enabled
   - Documentation comments for public APIs

## Security Considerations

1. **API Authentication**:
   - JWT-based authentication for API endpoints
   - Role-based access control for administrative functions
   - Rate limiting for public endpoints
   - Token refresh mechanism for session management

2. **Data Privacy**:
   - Vehicle data should be anonymized where possible
   - User consent required for data collection
   - GDPR compliance for European users
   - Data retention policies clearly defined

3. **Service-to-Service Communication**:
   - API keys for ML service communication
   - HTTPS for all external communications
   - Input validation on all services
   - Secrets management via environment variables

4. **Infrastructure Security**:
   - Database encryption at rest
   - Network security groups to restrict access
   - Regular security updates
   - Vulnerability scanning

## Getting Started with the ML Service

To begin developing the ML service that integrates with this backend:

1. **Environment Setup**:
   - Set up a Python environment with FastAPI or Flask for the API layer
   - Install TensorFlow or PyTorch for ML model development
   - Configure PostgreSQL for model metadata storage
   - Set up Docker for containerization

2. **API Implementation**:
   - Implement the API contract described above
   - Set up authentication with the main backend
   - Create health check endpoints
   - Implement logging and monitoring

3. **Model Development**:
   - Start with simple models based on DTC codes only
   - Gradually incorporate OBD parameter data for more sophisticated predictions
   - Train specialized models for popular Romanian vehicle makes
   - Implement model versioning and storage

4. **Integration Testing**:
   - Test connectivity with the main backend
   - Verify data format consistency
   - Validate prediction accuracy
   - Benchmark performance under load

5. **Deployment**:
   - Deploy as a containerized service for easy scaling
   - Implement CI/CD for automated testing and deployment
   - Set up monitoring and alerting
   - Establish backup and disaster recovery procedures

## Roadmap and Future Directions

For detailed implementation plans and future directions, refer to the `ML_SERVICE_ROADMAP.md` file, which includes:

1. **Week-by-Week Implementation Plan**:
   - Infrastructure setup
   - Data processing pipelines
   - Model development
   - Integration and testing

2. **Deep Learning Architectures**:
   - Encoder-decoder networks
   - CNN-LSTM hybrids
   - Transformer-based models
   - Vehicle-specific neural networks

3. **Romanian Market Focus**:
   - Vehicle models common in Romania
   - Adaptation for local driving conditions
   - Regional service center integration
   - Localized recommendations

## References

- For DTC code formats and meanings, refer to standard OBD-II specifications
- For ML model inspiration, see the machine learning projects listed in the resources document
- For dataset access, check the links provided in `AUTOMOTIVE_DATASETS_AND_RESOURCES.md`
- For implementation timeline and technical details, see `ML_SERVICE_ROADMAP.md` 