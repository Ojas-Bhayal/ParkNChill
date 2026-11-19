package com.parknchillai.ai;

/**
 * Typed exception thrown when a dataset or serialized model has an incompatible
 * schema ID or enhancement policy ID (WS-6 requirement).
 */
public class SchemaMismatchException extends IllegalArgumentException {

    private final String expectedSchema;
    private final String actualSchema;
    private final String expectedPolicy;
    private final String actualPolicy;

    public SchemaMismatchException(String expectedSchema, String actualSchema,
                                   String expectedPolicy, String actualPolicy,
                                   String message) {
        super(message);
        this.expectedSchema = expectedSchema;
        this.actualSchema = actualSchema;
        this.expectedPolicy = expectedPolicy;
        this.actualPolicy = actualPolicy;
    }

    public String getExpectedSchema() { return expectedSchema; }
    public String getActualSchema()   { return actualSchema; }
    public String getExpectedPolicy() { return expectedPolicy; }
    public String getActualPolicy()   { return actualPolicy; }
}
